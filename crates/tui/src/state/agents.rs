//! Subagent, process, and background-task mutations on `AppState`.
use super::*;

impl AppState {
    pub fn running_agent_count(&self) -> usize {
        self.agent_rows.iter().filter(|r| r.is_live()).count()
    }

    pub fn running_process_count(&self) -> usize {
        self.agent_rows
            .iter()
            .filter(|r| r.is_running_process())
            .count()
    }

    pub fn descendant_agent_ids(&self, id: &str) -> Vec<String> {
        let mut out = vec![id.to_string()];
        let mut grew = true;
        while grew {
            grew = false;
            for row in &self.agent_rows {
                if !row.is_subagent() {
                    continue;
                }
                let Some(parent) = row.parent_id.as_deref() else {
                    continue;
                };
                if out.iter().any(|p| p == parent) && !out.iter().any(|p| p == &row.id) {
                    out.push(row.id.clone());
                    grew = true;
                }
            }
        }
        out
    }

    pub fn sort_agents(&mut self) {
        self.agent_rows
            .sort_by(|a, b| match (a.kind.as_str(), b.kind.as_str()) {
                ("subagent", "process") => std::cmp::Ordering::Less,
                ("process", "subagent") => std::cmp::Ordering::Greater,
                _ => a
                    .depth
                    .cmp(&b.depth)
                    .then(a.index.cmp(&b.index))
                    .then(a.id.cmp(&b.id)),
            });
    }

    pub fn nudge_agents(&mut self) {
        if self.agents_nudged {
            return;
        }
        self.agents_nudged = true;
        self.set_toast("/agents to monitor");
    }

    pub fn merge_agent_snapshot(
        &mut self,
        processes: &serde_json::Value,
        status: &serde_json::Value,
    ) {
        if self.agents_replay {
            return;
        }
        let snap = parse_agent_rows(processes, status);
        let prior_output: Vec<(String, String)> = self
            .agent_rows
            .iter()
            .filter(|r| r.is_process() && !r.output.is_empty())
            .map(|r| (r.id.clone(), r.output.clone()))
            .collect();
        self.agent_rows.retain(|r| r.kind != "process");
        for mut row in snap.iter().filter(|r| r.is_process()).cloned() {
            if let Some((_, out)) = prior_output.iter().find(|(id, _)| id == &row.id) {
                if row.output.len() < out.len() {
                    row.output = out.clone();
                }
            }
            self.agent_rows.push(row);
        }
        for row in snap.iter().filter(|r| r.is_subagent()) {
            if let Some(existing) = self.agent_rows.iter_mut().find(|r| r.id == row.id) {
                if !is_terminal_agent_status(&existing.status) {
                    existing.status = row.status.clone();
                }
                existing.depth = row.depth;
                existing.parent_id = row.parent_id.clone();
                existing.index = row.index;
                if !row.title.is_empty() && row.title != "subagent" {
                    existing.title = row.title.clone();
                }
                if !row.model.is_empty() {
                    existing.model = row.model.clone();
                }
                if row.tool_count > existing.tool_count {
                    existing.tool_count = row.tool_count;
                }
                existing.extra = row.extra.clone();
            } else {
                self.agent_rows.push(row.clone());
            }
        }
        self.sort_agents();
        self.mark_dirty();
    }

    pub fn subagent_payload_id(payload: &serde_json::Value) -> String {
        payload
            .get("subagent_id")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .unwrap_or_else(|| {
                let idx = payload
                    .get("task_index")
                    .and_then(|x| x.as_u64())
                    .unwrap_or(0);
                let goal = payload
                    .get("goal")
                    .and_then(|x| x.as_str())
                    .unwrap_or("subagent");
                format!("sa:{idx}:{goal}")
            })
    }

    pub fn upsert_subagent(
        &mut self,
        payload: &serde_json::Value,
        status_hint: Option<&str>,
        create: bool,
    ) -> bool {
        let id = Self::subagent_payload_id(payload);
        let existing = self
            .agent_rows
            .iter()
            .position(|r| r.id == id && r.is_subagent());
        if existing.is_none() && !create {
            return false;
        }
        if existing.is_none() {
            self.agent_rows.push(AgentRow::subagent(id.clone()));
        }
        let Some(row) = self
            .agent_rows
            .iter_mut()
            .find(|r| r.id == id && r.is_subagent())
        else {
            return false;
        };
        if let Some(goal) = payload
            .get("goal")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
        {
            row.title = goal.to_string();
        }
        if let Some(hint) = status_hint {
            if is_terminal_agent_status(hint) || !is_terminal_agent_status(&row.status) {
                row.status = hint.to_string();
            }
        } else if let Some(st) = payload.get("status").and_then(|x| x.as_str()) {
            row.status = st.to_string();
        }
        if let Some(d) = payload.get("depth").and_then(|x| x.as_u64()) {
            row.depth = d as u32;
        }
        if let Some(p) = payload.get("parent_id").and_then(|x| x.as_str()) {
            row.parent_id = if p.is_empty() {
                None
            } else {
                Some(p.to_string())
            };
        }
        if let Some(m) = payload
            .get("model")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
        {
            row.model = m.to_string();
        }
        if let Some(n) = json_u64(payload, "tool_count") {
            row.tool_count = row.tool_count.max(n);
        }
        if let Some(n) = json_u64(payload, "input_tokens") {
            row.input_tokens = row.input_tokens.max(n);
        }
        if let Some(n) = json_u64(payload, "output_tokens") {
            row.output_tokens = row.output_tokens.max(n);
        }
        if let Some(n) = json_u64(payload, "api_calls") {
            row.api_calls = row.api_calls.max(n);
        }
        if let Some(c) = json_f64(payload, "cost_usd") {
            if c > row.cost_usd {
                row.cost_usd = c;
            }
        }
        if let Some(n) = json_u64(payload, "iteration") {
            row.iteration = row.iteration.max(n);
        }
        if let Some(idx) = payload.get("task_index").and_then(|x| x.as_u64()) {
            row.index = idx;
        }
        if let Some(secs) = payload
            .get("duration_seconds")
            .and_then(|x| x.as_f64())
            .or_else(|| {
                payload
                    .get("duration_seconds")
                    .and_then(|x| x.as_u64())
                    .map(|n| n as f64)
            })
        {
            row.duration_secs = Some(secs);
        }
        let summary = payload
            .get("summary")
            .and_then(|x| x.as_str())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                if is_terminal_agent_status(&row.status) {
                    payload
                        .get("text")
                        .and_then(|x| x.as_str())
                        .filter(|s| !s.is_empty())
                } else {
                    None
                }
            });
        if let Some(sum) = summary {
            row.summary = sum.to_string();
        }
        row.extra = format!("d{} {}", row.depth, row.model).trim().to_string();
        self.sort_agents();
        self.mark_dirty();
        true
    }

    pub fn push_agent_note(&mut self, id: &str, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        if let Some(row) = self
            .agent_rows
            .iter_mut()
            .find(|r| r.id == id && r.is_subagent())
        {
            push_capped(&mut row.notes, line.trim().to_string());
            self.mark_dirty();
        }
    }

    pub fn push_agent_thought(&mut self, id: &str, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        if let Some(row) = self
            .agent_rows
            .iter_mut()
            .find(|r| r.id == id && r.is_subagent())
        {
            push_capped(&mut row.thinking, line.trim().to_string());
            self.mark_dirty();
        }
    }

    pub fn append_process_output(&mut self, id: &str, chunk: &str) {
        if chunk.is_empty() {
            return;
        }
        if let Some(row) = self
            .agent_rows
            .iter_mut()
            .find(|r| r.id == id && r.is_process())
        {
            row.output.push_str(chunk);
            const CAP: usize = 4000;
            if row.output.len() > CAP {
                let extra = row.output.len() - CAP;
                row.output.drain(..extra);
            }
            if row.status.is_empty() {
                row.status = "running".into();
            }
            self.mark_dirty();
        } else {
            let mut row = AgentRow::subagent(id.to_string());
            row.kind = "process".into();
            row.title = "process".into();
            row.status = "running".into();
            row.output = chunk.to_string();
            self.agent_rows.push(row);
            self.sort_agents();
            self.mark_dirty();
        }
    }

    pub fn close_process(&mut self, id: &str) {
        if let Some(row) = self
            .agent_rows
            .iter_mut()
            .find(|r| r.id == id && r.is_process())
        {
            row.status = "exited".into();
            self.mark_dirty();
        }
    }

    pub fn push_agent_tool(&mut self, id: &str, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        if let Some(row) = self
            .agent_rows
            .iter_mut()
            .find(|r| r.id == id && r.is_subagent())
        {
            row.last_tool = line.trim().to_string();
            row.tool_count = row.tool_count.saturating_add(1);
            self.mark_dirty();
        }
    }

    pub fn running_bg_count(&self) -> usize {
        self.bg_tasks
            .iter()
            .filter(|t| t.status == BgStatus::Running)
            .count()
    }

    pub fn open_background(&mut self) {
        self.active_view = ActiveView::Background;
        self.modal_selected = 0;
        self.picker_filter.clear();
        self.picker_list = None;
        self.mark_dirty();
    }

    pub fn start_bg_task(&mut self, id: String, prompt: String) {
        self.bg_tasks.retain(|t| t.id != id);
        self.bg_tasks.insert(
            0,
            BgTask {
                id,
                prompt,
                status: BgStatus::Running,
                result: String::new(),
                started: Instant::now(),
            },
        );
        self.prune_bg_tasks();
        self.mark_dirty();
    }

    pub fn complete_bg_task(&mut self, id: &str, text: &str) {
        if let Some(t) = self.bg_tasks.iter_mut().find(|t| t.id == id) {
            t.status = BgStatus::Done;
            t.result = text.to_string();
        } else {
            self.bg_tasks.insert(
                0,
                BgTask {
                    id: id.to_string(),
                    prompt: String::new(),
                    status: BgStatus::Done,
                    result: text.to_string(),
                    started: Instant::now(),
                },
            );
        }
        self.prune_bg_tasks();
        if self.active_view == ActiveView::Background {
            self.clamp_modal(self.picker_len());
        }
        self.mark_dirty();
    }

    fn prune_bg_tasks(&mut self) {
        const KEEP: usize = 24;
        while self.bg_tasks.len() > KEEP {
            match self
                .bg_tasks
                .iter()
                .rposition(|t| t.status == BgStatus::Done)
            {
                Some(i) => {
                    self.bg_tasks.remove(i);
                }
                None => break,
            }
        }
    }

    pub fn filtered_bg_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.bg_tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                Self::filter_matches(&t.id, q)
                    || Self::filter_matches(&t.prompt, q)
                    || Self::filter_matches(&t.result, q)
            })
            .map(|(i, _)| i)
            .collect()
    }
}
