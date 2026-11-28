//! Rust port of `apps/shared/src/json-rpc-gateway.ts` replay state.
//!
//! Keep the upstream distinction: ordinary live events always dispatch (their
//! cursor only advances); replay and held live events dispatch only if newer.
//! Process epochs must never erase an unrelated canonical session's cursor.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};

pub const REPLAY_CURSOR_MAX: usize = 256;
pub const REPLAY_REQUEST_TIMEOUT_MS: u64 = 10_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReplayAction {
    Dispatch {
        event: Value,
    },
    Request {
        session_id: String,
        params: Value,
        generation: u64,
    },
    Barrier {
        session_id: String,
        generation: u64,
        valid: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierState {
    Pending(u64),
    Invalid,
    Absent,
}

#[derive(Debug, Default)]
struct Hold {
    events: VecDeque<Value>,
}
#[derive(Debug, Default)]
struct Holds(Vec<(String, Hold)>);
impl Holds {
    fn insert(&mut self, id: String, hold: Hold) {
        self.0.push((id, hold));
    }
    fn contains_key(&self, id: &str) -> bool {
        self.0.iter().any(|(key, _)| key == id)
    }
    fn get_mut(&mut self, id: &str) -> Option<&mut Hold> {
        self.0
            .iter_mut()
            .find(|(key, _)| key == id)
            .map(|(_, hold)| hold)
    }
    fn remove(&mut self, id: &str) -> Option<Hold> {
        let index = self.0.iter().position(|(key, _)| key == id)?;
        Some(self.0.remove(index).1)
    }
    fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.iter().map(|(key, _)| key)
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn into_values(self) -> impl Iterator<Item = Hold> {
        self.0.into_iter().map(|(_, hold)| hold)
    }
}

/// Transport-independent state port. Actions are consumed in returned order.
#[derive(Debug)]
pub struct ReplayMachine {
    enabled: bool,
    open: bool,
    cursors: VecDeque<(String, f64)>,
    process_epoch: Option<String>,
    session_epochs: HashMap<String, String>,
    generation: u64,
    hold: Option<Holds>,
}

impl Default for ReplayMachine {
    fn default() -> Self {
        Self::new(true)
    }
}
impl ReplayMachine {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            open: false,
            cursors: VecDeque::new(),
            process_epoch: None,
            session_epochs: HashMap::new(),
            generation: 0,
            hold: None,
        }
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn watermarks(&self) -> Value {
        self.cursors
            .iter()
            .map(|(sid, seq)| (sid.clone(), number(*seq)))
            .collect::<serde_json::Map<_, _>>()
            .into()
    }
    pub fn session_epochs(&self) -> Value {
        json!(self.session_epochs)
    }
    pub fn barrier(&self, sid: &str) -> BarrierState {
        if self
            .hold
            .as_ref()
            .is_some_and(|hold| hold.contains_key(sid))
        {
            BarrierState::Pending(self.generation)
        } else if self.enabled && !self.open && self.cursor(sid).is_some() {
            BarrierState::Invalid
        } else {
            BarrierState::Absent
        }
    }
    pub fn opened(&mut self) -> Vec<ReplayAction> {
        self.open = true;
        if !self.enabled || self.hold.is_some() || self.cursors.is_empty() {
            return vec![];
        }
        self.generation += 1;
        let mut hold = Holds::default();
        let requests = self
            .cursors
            .iter()
            .map(|(sid, seq)| {
                hold.insert(sid.clone(), Hold::default());
                let mut params = json!({"session_id":sid,"last_seen":number(*seq)});
                if let Some(epoch) = self.session_epochs.get(sid) {
                    params["replay_epoch"] = json!(epoch);
                }
                ReplayAction::Request {
                    session_id: sid.clone(),
                    params,
                    generation: self.generation,
                }
            })
            .collect();
        self.hold = Some(hold);
        requests
    }
    pub fn closed(&mut self) -> Vec<ReplayAction> {
        self.open = false;
        self.cancel(false).1
    }
    pub fn live(&mut self, event: Value) -> Vec<ReplayAction> {
        let mut actions = Vec::new();
        if event["type"] == "gateway.ready"
            && let Some(epoch) = nonempty(&event["payload"]["replay_epoch"])
        {
            self.adopt_process_epoch(epoch.to_owned(), &mut actions);
        }
        let sid = nonempty(&event["session_id"]).map(str::to_owned);
        if let Some(sid) = &sid
            && sequence(&event).is_some()
            && let Some(hold) = self.hold.as_mut().and_then(|hold| hold.get_mut(sid))
        {
            hold.events.push_back(event);
            return actions;
        }
        if let Some(sid) = sid {
            self.adopt_session_epoch(&sid, event.get("replay_epoch"));
            if let Some(seq) = sequence(&event)
                && seq > self.cursor(&sid).unwrap_or(0.0)
            {
                self.advance(&sid, seq, &mut actions);
            }
        }
        self.dispatch(event, &mut actions);
        actions
    }
    /// An old socket's late reply cannot consume a replacement replay hold.
    pub fn completed(&mut self, sid: &str, generation: u64, result: Value) -> Vec<ReplayAction> {
        if generation != self.generation
            || self
                .hold
                .as_ref()
                .is_none_or(|hold| !hold.contains_key(sid))
        {
            return vec![];
        }
        let mut actions = Vec::new();
        let skip = self.apply_epochs(sid, &result, &mut actions);
        if !skip && result["truncated"] == true {
            let mut head = self.cursor(sid).unwrap_or(0.0);
            if let Some(latest) = result["latest_seq"].as_f64().filter(|n| n.is_finite()) {
                head = head.max(latest);
            }
            for event in result["events"].as_array().into_iter().flatten() {
                if let Some(seq) = sequence(event) {
                    head = head.max(seq);
                }
            }
            self.advance(sid, head, &mut actions);
        } else if !skip {
            for event in result["events"].as_array().into_iter().flatten() {
                if event["type"].as_str().is_some_and(|s| !s.is_empty()) {
                    self.dispatch_newer(replayed(event.clone()), &mut actions);
                }
            }
        }
        if generation == self.generation {
            self.flush(sid, generation, &mut actions);
        }
        actions
    }
    /// Unsupported/timeout failures release only their own session. not_found
    /// also retires the cursor and its qualifying session epoch.
    pub fn failed(
        &mut self,
        sid: &str,
        generation: u64,
        reason: Option<&str>,
    ) -> Vec<ReplayAction> {
        if generation != self.generation
            || self
                .hold
                .as_ref()
                .is_none_or(|hold| !hold.contains_key(sid))
        {
            return vec![];
        }
        if reason == Some("not_found") {
            self.forget(sid);
        }
        let mut actions = Vec::new();
        self.flush(sid, generation, &mut actions);
        actions
    }
    fn cursor(&self, sid: &str) -> Option<f64> {
        self.cursors
            .iter()
            .find(|(id, _)| id == sid)
            .map(|(_, seq)| *seq)
    }
    fn forget(&mut self, sid: &str) {
        self.cursors.retain(|(id, _)| id != sid);
        self.session_epochs.remove(sid);
    }
    fn advance(&mut self, sid: &str, seq: f64, actions: &mut Vec<ReplayAction>) {
        self.cursors.retain(|(id, _)| id != sid);
        self.cursors.push_back((sid.into(), seq));
        while self.cursors.len() > REPLAY_CURSOR_MAX {
            let Some(index) = self
                .cursors
                .iter()
                .position(|(id, _)| self.hold.as_ref().is_none_or(|hold| !hold.contains_key(id)))
            else {
                break;
            };
            let (id, seq) = self.cursors.remove(index).expect("cursor index exists");
            let epoch = self.session_epochs.remove(&id);
            let mut payload = json!({"latest_seq":number(seq)});
            if let Some(epoch) = epoch {
                payload["replay_epoch"] = json!(epoch);
            }
            self.dispatch(
                json!({"type":"session.replay_gap","session_id":id,"payload":payload}),
                actions,
            );
        }
    }
    fn dispatch(&self, mut event: Value, actions: &mut Vec<ReplayAction>) {
        if let Some(epoch) = &self.process_epoch {
            event["replayEpoch"] = json!(epoch);
        }
        actions.push(ReplayAction::Dispatch { event });
    }
    fn dispatch_newer(&mut self, event: Value, actions: &mut Vec<ReplayAction>) {
        if let Some(sid) = nonempty(&event["session_id"]).map(str::to_owned) {
            self.adopt_session_epoch(&sid, event.get("replay_epoch"));
            if let Some(seq) = sequence(&event) {
                if seq <= self.cursor(&sid).unwrap_or(0.0) {
                    return;
                }
                self.advance(&sid, seq, actions);
            }
        }
        self.dispatch(event, actions);
    }
    fn adopt_session_epoch(&mut self, sid: &str, epoch: Option<&Value>) -> bool {
        let Some(epoch) = epoch.and_then(nonempty) else {
            return false;
        };
        let changed = self.session_epochs.get(sid).is_some_and(|old| old != epoch);
        if changed {
            self.cursors.retain(|(id, _)| id != sid);
        }
        self.session_epochs.insert(sid.into(), epoch.into());
        changed
    }
    fn cancel(&mut self, valid: bool) -> (Option<Holds>, Vec<ReplayAction>) {
        let hold = self.hold.take();
        let old_generation = self.generation;
        self.generation += 1;
        let actions = hold
            .as_ref()
            .into_iter()
            .flat_map(|hold| hold.keys())
            .map(|sid| ReplayAction::Barrier {
                session_id: sid.clone(),
                generation: old_generation,
                valid,
            })
            .collect();
        (hold, actions)
    }
    fn adopt_process_epoch(&mut self, epoch: String, actions: &mut Vec<ReplayAction>) {
        if self.process_epoch.as_ref() == Some(&epoch) {
            return;
        }
        let changed = self.process_epoch.is_some();
        self.process_epoch = Some(epoch);
        if changed {
            self.cursors
                .retain(|(id, _)| self.session_epochs.contains_key(id));
            let (hold, barriers) = self.cancel(true);
            for hold in hold.into_iter().flat_map(Holds::into_values) {
                for event in hold.events {
                    self.dispatch_newer(replayed(event), actions);
                }
            }
            // Promise observers see parked frames before resolved barriers.
            actions.extend(barriers);
        }
    }
    fn apply_epochs(&mut self, sid: &str, result: &Value, actions: &mut Vec<ReplayAction>) -> bool {
        if result["snapshot_required"] == true {
            self.forget(sid);
            let mut payload = serde_json::Map::new();
            if let Some(epoch) = result.get("replay_epoch") {
                payload.insert("replay_epoch".into(), epoch.clone());
            }
            if let Some(seq) = result.get("latest_seq") {
                payload.insert("latest_seq".into(), seq.clone());
            }
            self.dispatch(
                json!({"type":"session.replay_gap","session_id":sid,"payload":payload}),
                actions,
            );
            return true;
        }
        if nonempty(&result["replay_epoch"]).is_some() {
            if self.adopt_session_epoch(sid, result.get("replay_epoch")) {
                return true;
            }
        } else if let Some(epoch) = nonempty(&result["epoch"]) {
            if self.process_epoch.as_ref().is_some_and(|old| old != epoch) {
                self.adopt_process_epoch(epoch.into(), actions);
                return true;
            }
            if self.process_epoch.is_none() {
                self.process_epoch = Some(epoch.into());
            }
        }
        false
    }
    fn flush(&mut self, sid: &str, generation: u64, actions: &mut Vec<ReplayAction>) {
        loop {
            let next = self
                .hold
                .as_mut()
                .and_then(|hold| hold.get_mut(sid))
                .and_then(|hold| hold.events.pop_front());
            let Some(event) = next else {
                break;
            };
            self.dispatch_newer(replayed(event), actions);
        }
        let existed = self
            .hold
            .as_mut()
            .and_then(|hold| hold.remove(sid))
            .is_some();
        if self.hold.as_ref().is_some_and(Holds::is_empty) {
            self.hold = None;
        }
        if existed {
            actions.push(ReplayAction::Barrier {
                session_id: sid.into(),
                generation,
                valid: true,
            });
        }
    }
}
fn nonempty(value: &Value) -> Option<&str> {
    value.as_str().filter(|s| !s.is_empty())
}
fn sequence(event: &Value) -> Option<f64> {
    event["seq"].as_f64().filter(|n| n.is_finite())
}
fn number(value: f64) -> Value {
    if value.fract() == 0.0 && value >= 0.0 && value <= u64::MAX as f64 {
        json!(value as u64)
    } else if value.fract() == 0.0 && value >= i64::MIN as f64 && value <= i64::MAX as f64 {
        json!(value as i64)
    } else {
        json!(value)
    }
}
fn replayed(mut event: Value) -> Value {
    event["replayed"] = json!(true);
    event
}
