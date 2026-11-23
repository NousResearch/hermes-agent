//! Durable follow-up ownership for the native engine. Stop parks rather than
//! deletes; recovered work is never silently submitted after a restart.
use hermes_engine::{Attachment, History};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct QueuedPrompt { pub id: u64, pub text: String, #[serde(default)] pub attachments: Vec<Attachment> }

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct InFlight { pub entry: QueuedPrompt, pub turn_index: usize }

#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub(super) struct PromptQueue {
    pub entries: Vec<QueuedPrompt>,
    pub parked: bool,
    pub in_flight: Option<InFlight>,
    next_id: u64,
}

impl PromptQueue {
    pub fn load(directory: &Path, history: &History) -> Result<Self, String> {
        let mut queue: Self = match fs::read(directory.join("queued-prompts.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| format!("Could not read queued messages: {e}"))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(e) => return Err(e.to_string()),
        };
        queue.settle(history);
        if !queue.entries.is_empty() { queue.parked = true; }
        queue.save(directory)?;
        Ok(queue)
    }
    pub fn save(&self, directory: &Path) -> Result<(), String> {
        let path = directory.join("queued-prompts.json");
        if self.entries.is_empty() && self.in_flight.is_none() {
            return match fs::remove_file(path) { Ok(()) => Ok(()), Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()), Err(e) => Err(e.to_string()) };
        }
        let temp = directory.join("queued-prompts.tmp");
        fs::write(&temp, serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        fs::rename(temp, path).map_err(|e| e.to_string())
    }
    pub fn push(&mut self, text: String, attachments: Vec<Attachment>, front: bool) -> Result<u64, String> {
        Attachment::validate_set(&attachments).map_err(|e|e.to_string())?;
        if text.trim().is_empty() && attachments.is_empty() { return Err("Enter a message to queue.".into()); }
        if self.entries.len() >= 50 { return Err("The queue is full. Send or remove a queued message first.".into()); }
        if text.len() > 128 * 1024 { return Err("A queued message can contain at most 128 KiB of text.".into()); }
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let entry = QueuedPrompt { id: self.next_id, text, attachments };
        let id = entry.id;
        if front { self.entries.insert(0, entry); } else { self.entries.push(entry); }
        Ok(id)
    }
    pub fn begin(&mut self, text: String, attachments: Vec<Attachment>, turn_index: usize) {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        self.in_flight = Some(InFlight { entry: QueuedPrompt { id: self.next_id, text, attachments }, turn_index });
    }
    pub fn begin_next(&mut self, turn_index: usize) -> Option<String> {
        if self.parked || self.entries.is_empty() { return None; }
        let entry = self.entries.remove(0);
        let text = entry.text.clone();
        self.in_flight = Some(InFlight { entry, turn_index });
        Some(text)
    }
    pub fn admitted(&mut self, history: &History) -> bool {
        if self.in_flight.as_ref().is_some_and(|pending| history.turns.get(pending.turn_index).is_some_and(|turn| turn.prompt == pending.entry.text && turn.attachments == pending.entry.attachments)) {
            self.in_flight = None; true
        } else { false }
    }
    pub fn settle(&mut self, history: &History) {
        if !self.admitted(history) && let Some(pending) = self.in_flight.take() {
            self.entries.insert(0, pending.entry); self.parked = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hermes_engine::{Status, Turn};

    #[test]
    fn recovery_keeps_unadmitted_messages_and_parks_queue() {
        let directory = tempfile::tempdir().unwrap();
        let mut queue = PromptQueue::default();
        queue.push("second".into(), vec![], false).unwrap();
        queue.begin("first".into(), vec![], 0);
        queue.save(directory.path()).unwrap();
        let restored = PromptQueue::load(directory.path(), &History::default()).unwrap();
        assert!(restored.parked);
        assert_eq!(restored.entries.iter().map(|entry| entry.text.as_str()).collect::<Vec<_>>(), ["first", "second"]);
    }

    #[test]
    fn recovery_does_not_replay_already_recorded_turn() {
        let directory = tempfile::tempdir().unwrap();
        let mut queue = PromptQueue::default();
        queue.begin("recorded".into(), vec![], 0);
        queue.push("next".into(), vec![], false).unwrap();
        queue.save(directory.path()).unwrap();
        let mut history = History::default();
        history.turns.push(Turn { context: String::new(), attachments: vec![], prompt:"recorded".into(), model:"test".into(), entries:vec![], status:Status::Interrupted });
        let restored = PromptQueue::load(directory.path(), &history).unwrap();
        assert_eq!(restored.entries.len(), 1);
        assert_eq!(restored.entries[0].text, "next");
        assert!(restored.in_flight.is_none());
    }

    #[test]
    fn parked_queue_is_not_admitted_until_resumed() {
        let mut queue = PromptQueue::default(); queue.push("follow-up".into(), vec![], false).unwrap();
        queue.parked = true; assert!(queue.begin_next(0).is_none());
        queue.parked = false; assert_eq!(queue.begin_next(0).as_deref(), Some("follow-up"));
    }
}
