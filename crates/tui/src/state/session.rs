//! Session, queue, transcript, and agent mutations on `AppState`.
use super::*;
use chrono::Local;

impl AppState {
    pub fn enqueue(&mut self, text: String) {
        self.prompt_queue.push_back(text);
        self.mark_dirty();
    }

    pub fn take_queued(&mut self) -> Option<String> {
        let out = self.prompt_queue.pop_front()?;
        if let Some(i) = self.queue_edit {
            self.queue_edit = if i == 0 { None } else { Some(i - 1) };
        }
        self.mark_dirty();
        Some(out)
    }

    pub fn cycle_queue(&mut self, dir: i32) -> Option<String> {
        let n = self.prompt_queue.len();
        if n == 0 {
            self.queue_edit = None;
            return None;
        }
        let next = match self.queue_edit {
            None => {
                if dir > 0 {
                    0
                } else {
                    n - 1
                }
            }
            Some(i) => ((i as i32 + dir).rem_euclid(n as i32)) as usize,
        };
        self.queue_edit = Some(next);
        self.mark_dirty();
        self.prompt_queue.get(next).cloned()
    }

    pub fn cancel_queue_edit(&mut self) {
        self.queue_edit = None;
        self.mark_dirty();
    }

    pub fn drop_queue_edit(&mut self) -> bool {
        let Some(i) = self.queue_edit.take() else {
            return false;
        };
        self.drop_queued(i)
    }

    pub fn drop_queued(&mut self, i: usize) -> bool {
        if i >= self.prompt_queue.len() {
            return false;
        }
        self.prompt_queue.remove(i);
        if let Some(e) = self.queue_edit {
            if e == i {
                self.queue_edit = None;
            } else if e > i {
                self.queue_edit = Some(e - 1);
            }
        }
        self.mark_dirty();
        true
    }

    pub fn take_queue_edit(&mut self) -> Option<String> {
        let i = self.queue_edit.take()?;
        self.mark_dirty();
        self.prompt_queue.remove(i)
    }

    pub fn reset_session(&mut self, session_id: String) {
        self.session_id = Some(session_id);
        self.session_key.clear();
        self.messages.clear();
        self.expanded_tools.clear();
        self.expand_epoch = self.expand_epoch.wrapping_add(1);
        self.hit_tools.clear();
        self.tasks.clear();
        self.prompt_queue.clear();
        self.queue_edit = None;
        self.trace_selected = 0;
        self.trace_focus = false;
        self.trace_follow = true;
        self.trace_open = false;
        self.resume_step = None;
        self.pending_approval = None;
        self.pending_clarify = None;
        self.pending_secret = None;
        self.bg_tasks.clear();
        self.agent_rows.clear();
        self.agents_steer = false;
        self.agents_replay = false;
        self.agents_nudged = false;
        self.is_generating = false;
        self.scroll_from_bottom = 0;
        self.goal = None;
        self.metrics.activity.clear();
        self.metrics.turn_start_time = None;
        self.intro_tools.clear();
        self.intro_skills.clear();
        self.mcp_connected = 0;
        self.mcp_servers.clear();
        self.shell_context.clear();
        self.metrics.permission_mode = PermissionMode::Manual;
        self.intro_warning = None;
        self.session_title = "Hermes TUI".into();
        self.session_started = Instant::now();
        self.session_ready = false;
        self.reveal_started = None;
        self.metrics.context_used = 0;
        self.metrics.total_tokens = 0;
        self.mark_dirty();
    }

    pub fn display_title(&self) -> String {
        self.title_for_width(28)
    }

    fn title_for_width(&self, width: usize) -> String {
        let clip = |s: &str| crate::tips::ellipsize(s, width);
        let t = self.session_title.trim();
        if !t.is_empty() && t != "Hermes TUI" {
            return clip(t);
        }
        if let Some(user) = self.messages.iter().find(|m| m.role == MessageRole::User) {
            let line = user.content.lines().next().unwrap_or("").trim();
            if !line.is_empty() {
                return clip(line);
            }
        }
        clip(
            self.metrics
                .cwd
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or("session"),
        )
    }

    /// Ghostty / iTerm tab title (OSC 0). Mirrors Grok: `activity - session - hermes`.
    pub fn tab_title(&self) -> String {
        let session = self.title_for_width(48);
        let mut parts: Vec<String> = Vec::new();
        if self.pending_approval.is_some() {
            parts.push("approval needed".into());
        } else if self.pending_secret.is_some() || self.pending_clarify.is_some() {
            parts.push("waiting for input".into());
        } else if self.show_turn_bar() {
            parts.push(self.tab_activity());
        }
        if !session.is_empty() && parts.last().map(|p| p.as_str()) != Some(session.as_str()) {
            parts.push(session);
        }
        parts.push("hermes".into());
        sanitize_tab_title(&parts.join(" - "))
    }

    fn tab_activity(&self) -> String {
        format!("{}…", self.live_status())
    }

    /// Ink FaceTicker copy: rotating wait verb, or `compacting`.
    pub fn live_status(&self) -> String {
        if self.metrics.is_compacting {
            return "compacting".into();
        }
        if self.pending_approval.is_some() {
            return "Waiting for approval".into();
        }
        if self.pending_secret.is_some() || self.pending_clarify.is_some() {
            return "Waiting for input".into();
        }
        wait_status_label(&self.metrics.activity)
    }

    fn last_running_tool(&self) -> Option<&ChatMessage> {
        self.messages.iter().rev().find(|m| {
            matches!(
                &m.role,
                MessageRole::Tool { status, .. } if status.contains("running")
            )
        })
    }

    pub fn freeze_thought(&mut self) {
        if let Some(idx) = self.last_turn_reasoning_index() {
            self.messages[idx].is_streaming = false;
            self.mark_dirty();
        }
    }

    pub fn needs_animation(&self) -> bool {
        self.messages.is_empty()
            || self.show_turn_bar()
            || self.flash_kind().is_some()
            || self.reveal() < 1.0
            || self
                .metrics
                .toast_message
                .as_ref()
                .is_some_and(|t| t.live())
            || (self.active_view == ActiveView::Background && self.running_bg_count() > 0)
            || self.running_agent_count() > 0
            || self.running_process_count() > 0
    }

    const BOOT_MS: u128 = 2500;
    const REVEAL_MS: u128 = 720;

    pub fn mark_session_ready(&mut self) {
        if self.session_ready {
            return;
        }
        self.session_ready = true;
        self.reveal_started = Some(Instant::now());
        self.mark_dirty();
    }

    pub fn advance_boot(&mut self) {
        if !self.session_ready && self.session_started.elapsed().as_millis() >= Self::BOOT_MS {
            self.mark_session_ready();
        }
    }

    /// 0 = logo only, 1 = full chrome. Ease-out.
    pub fn reveal(&self) -> f32 {
        if !self.session_ready {
            return 0.0;
        }
        let Some(t0) = self.reveal_started else {
            return 1.0;
        };
        if crate::ui::motion::reduced_motion() {
            return 1.0;
        }
        let t = (t0.elapsed().as_millis() as f32 / Self::REVEAL_MS as f32).clamp(0.0, 1.0);
        1.0 - (1.0 - t) * (1.0 - t)
    }

    pub fn flash_kind(&self) -> Option<HoverKind> {
        let (kind, at) = self.click_flash.as_ref()?;
        if at.elapsed().as_millis() < 180 {
            Some(kind.clone())
        } else {
            None
        }
    }

    pub fn set_hover(&mut self, kind: HoverKind) -> bool {
        if self.hover == kind {
            return false;
        }
        self.hover = kind;
        self.mark_dirty();
        true
    }

    pub fn ping_click(&mut self, kind: HoverKind) {
        self.click_flash = Some((kind, Instant::now()));
        self.mark_dirty();
    }

    pub fn toggle_tool_expand(&mut self, id: &str) {
        if !self.expanded_tools.remove(id) {
            self.expanded_tools.insert(id.to_string());
        }
        self.expand_epoch = self.expand_epoch.wrapping_add(1);
        self.mark_dirty();
    }

    pub fn thought_expanded(&self, id: &str) -> bool {
        self.show_thinking || self.expanded_tools.contains(id)
    }

    pub fn rotate_tip_if_due(&mut self) {
        if !self.tips_open {
            return;
        }
        if self.tip_shown_at.elapsed().as_secs() >= crate::tips::ROTATE_SECS {
            self.next_tip();
        }
    }

    pub fn next_tip(&mut self) {
        let n = crate::tips::COUNT.max(1);
        self.tip_index = (self.tip_index + 1) % n;
        self.tip_shown_at = Instant::now();
        self.mark_dirty();
    }

    pub fn set_tips_open(&mut self, open: bool) {
        self.tips_open = open;
        crate::tips::save_open(&self.hermes_home, open);
        if open {
            self.tip_shown_at = Instant::now();
        } else {
            self.hit_tips_close = None;
            self.hit_tips_bar = None;
        }
        self.mark_dirty();
    }

    pub fn hover_at(&self, col: u16, row: u16) -> HoverKind {
        if self.hit_tips_close.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::TipsClose;
        }
        if self.hit_tips_bar.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::TipsBar;
        }
        if self.hit_mode.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Mode;
        }
        if self.hit_branch.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Branch;
        }
        if self.hit_model.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Model;
        }
        if self.hit_context.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Context;
        }
        if self.hit_session.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Session;
        }
        if self.hit_bg.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Background;
        }
        if self.hit_agents.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Agents;
        }
        if self.hit_process.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Process;
        }
        if let Some((_, entry)) = self
            .hit_dock_stop
            .iter()
            .find(|(h, _)| h.contains(col, row))
        {
            return HoverKind::DockStop(*entry);
        }
        if let Some((_, entry)) = self.hit_dock.iter().find(|(h, _)| h.contains(col, row)) {
            return HoverKind::Dock(*entry);
        }
        if self.hit_jump.is_some_and(|h| h.contains(col, row)) {
            return HoverKind::Jump;
        }
        if let Some((_, label)) = self.hit_pastes.iter().find(|(h, _)| h.contains(col, row)) {
            return HoverKind::Paste(label.clone());
        }
        if let Some((_, id)) = self.hit_tools.iter().find(|(h, _)| h.contains(col, row)) {
            return HoverKind::Tool(id.clone());
        }
        if let Some((_, kind)) = self.hit_queue.iter().find(|(h, _)| h.contains(col, row)) {
            return kind.clone();
        }
        if let Some(area) = self.queue_area {
            if col >= area.x
                && col < area.x.saturating_add(area.width)
                && row >= area.y
                && row < area.y.saturating_add(area.height)
            {
                let vis = row.saturating_sub(area.y) as usize;
                let n = self.prompt_queue.len();
                let (start, end) = crate::ui::queue::queue_window(n, self.queue_edit);
                let idx = start + vis;
                if idx < end && idx < n {
                    return HoverKind::Queue(idx);
                }
            }
        }
        if let Some(inner) = self.picker_list {
            if col >= inner.x
                && col < inner.x.saturating_add(inner.width)
                && row >= inner.y
                && row < inner.y.saturating_add(inner.height)
            {
                let idx = self.picker_offset + (row.saturating_sub(inner.y) as usize);
                if idx < self.picker_len() {
                    return HoverKind::Picker(idx);
                }
            }
        }
        if let Some(inner) = self.files_list {
            if col >= inner.x
                && col < inner.x.saturating_add(inner.width)
                && row >= inner.y
                && row < inner.y.saturating_add(inner.height)
            {
                let idx = self.files_offset + (row.saturating_sub(inner.y) as usize);
                if idx < self.files_rows.len() {
                    return HoverKind::Files(idx);
                }
            }
        }
        if let Some(inner) = self.work_list {
            if col >= inner.x
                && col < inner.x.saturating_add(inner.width)
                && row >= inner.y
                && row < inner.y.saturating_add(inner.height)
            {
                let idx = self.work_offset + (row.saturating_sub(inner.y) as usize);
                let n = if self.work_show_diff {
                    self.work_diff_files.len()
                } else {
                    self.agent_rows.len() + self.bg_tasks.len()
                };
                if idx < n {
                    return HoverKind::Work(idx);
                }
            }
        }
        if let Some(area) = self.composer_area {
            if col >= area.x
                && col < area.x.saturating_add(area.width)
                && row >= area.y
                && row < area.y.saturating_add(area.height)
            {
                return HoverKind::Composer;
            }
        }
        HoverKind::None
    }

    pub fn show_turn_bar(&self) -> bool {
        self.is_generating || self.metrics.is_compacting || self.metrics.active_tool.is_some()
    }

    pub fn arm(&mut self, kind: ArmedKind, toast: impl Into<String>) {
        self.armed = Some((kind, Instant::now() + Duration::from_secs(4)));
        self.set_toast(toast);
    }

    /// Consume a live arm of this kind. Expired or mismatched arms miss.
    pub fn take_armed(&mut self, kind: ArmedKind) -> bool {
        match self.armed {
            Some((k, until)) if k == kind && Instant::now() < until => {
                self.armed = None;
                true
            }
            Some((_, until)) if Instant::now() >= until => {
                self.armed = None;
                false
            }
            _ => false,
        }
    }

    pub fn has_unsaved(&self, draft: &str) -> bool {
        !draft.trim().is_empty() || !self.prompt_queue.is_empty()
    }

    pub fn has_thread(&self) -> bool {
        !self.messages.is_empty() || !self.prompt_queue.is_empty() || self.is_generating
    }

    pub fn close_complete(&mut self) {
        if self.complete_open || !self.complete_items.is_empty() {
            self.complete_open = false;
            self.complete_items.clear();
            self.complete_selected = 0;
            self.mark_dirty();
        }
    }

    pub fn set_complete(&mut self, items: Vec<CompleteItem>, replace_from: usize) {
        self.complete_items = items;
        self.complete_replace_from = replace_from;
        self.complete_selected = 0;
        self.complete_open = !self.complete_items.is_empty();
        self.mark_dirty();
    }

    pub fn remember_image(&mut self, path: PathBuf) {
        if self.pending_images.iter().any(|p| p.path == path) {
            return;
        }
        self.pending_images.push(pending_from_path(path));
        self.mark_dirty();
    }

    pub fn open_peek(&mut self, title: String, body: String, image: Option<PathBuf>) {
        self.peek_title = title;
        self.peek_body = body;
        self.peek_image = image;
        self.peek_offset = 0;
        self.active_view = ActiveView::Peek;
        self.mark_dirty();
    }

    pub fn open_bracket(&mut self, label: &str) -> bool {
        if let Some(chip) = self.paste_chips.iter().find(|c| c.label == label) {
            self.open_peek(chip.label.clone(), chip.body.clone(), None);
            return true;
        }
        if label.starts_with("[[ Image") {
            if let Some(img) = self.pending_images.first() {
                self.open_peek(
                    img.name.clone(),
                    img.path.display().to_string(),
                    Some(img.path.clone()),
                );
                return true;
            }
        }
        false
    }

    pub fn remember_paste(&mut self, body: String) -> String {
        let base = crate::paste::token_label(&body);
        let label = crate::paste::unique_label(&base, &self.paste_chips);
        self.paste_chips.push(crate::paste::PasteChip {
            label: label.clone(),
            body,
            path: None,
        });
        self.mark_dirty();
        label
    }

    pub fn refresh_pending_images(&mut self, text: &str) {
        let cwd = self.metrics.cwd.clone();
        let mut refs = crate::complete::image_refs_in(text, &cwd);
        if text.contains("[[ Image") {
            for old in &self.pending_images {
                if !refs.iter().any(|p| p == &old.path) {
                    refs.push(old.path.clone());
                }
            }
        }
        self.pending_images = refs.into_iter().map(pending_from_path).collect();
        let mut live = text.to_string();
        for q in &self.prompt_queue {
            live.push('\n');
            live.push_str(q);
        }
        crate::paste::prune(&mut self.paste_chips, &live);
        self.mark_dirty();
    }

    pub fn set_toast(&mut self, text: impl Into<String>) {
        self.set_toast_for(text, crate::optimistic::toast_ttl());
    }

    pub fn set_toast_for(&mut self, text: impl Into<String>, ttl: Duration) {
        self.metrics.toast_message = Some(Toast {
            text: text.into(),
            created: Instant::now(),
            ttl,
        });
        self.mark_dirty();
    }

    pub fn take_live_undo(&mut self) -> Option<PendingUndo> {
        let u = self.pending_undo.take()?;
        if u.live() {
            Some(u)
        } else {
            None
        }
    }

    pub fn apply_undo(&mut self) -> String {
        let Some(u) = self.take_live_undo() else {
            return "nothing to undo".into();
        };
        match u.kind {
            UndoKind::File { rel, previous } => {
                let result = match &previous {
                    Some(bytes) => {
                        crate::platform::write_worktree_bytes(&self.metrics.cwd, &rel, bytes)
                    }
                    None => {
                        match crate::platform::confined_worktree_path(&self.metrics.cwd, &rel) {
                            Ok(path) if path.exists() => {
                                std::fs::remove_file(&path).map_err(|e| e.to_string())
                            }
                            Ok(_) => Ok(()),
                            Err(e) => Err(e),
                        }
                    }
                };
                match result {
                    Ok(()) => {
                        self.refresh_files();
                        format!("undid restore · {rel}")
                    }
                    Err(e) => {
                        self.pending_undo = Some(PendingUndo {
                            kind: UndoKind::File { rel, previous },
                            created: u.created,
                        });
                        format!("undo failed: {e}")
                    }
                }
            }
            UndoKind::Transcript { messages } => {
                self.messages = messages;
                self.mark_dirty();
                "undid clear".into()
            }
        }
    }

    pub fn clear_transcript(&mut self) -> String {
        if self.messages.is_empty() {
            return "already empty".into();
        }
        let messages = std::mem::take(&mut self.messages);
        self.pending_undo = Some(PendingUndo {
            kind: UndoKind::Transcript { messages },
            created: Instant::now(),
        });
        self.mark_dirty();
        "cleared · u undo".into()
    }

    pub fn begin_compaction(&mut self) {
        self.metrics.is_compacting = true;
        self.metrics.compaction_status = "folding context".into();
        self.metrics.compaction_started = Some(Instant::now());
        self.metrics.compaction_hide_at = None;
        self.metrics.compaction_painted = false;
        self.pending_fold = None;
        self.mark_dirty();
    }

    pub fn end_compaction(&mut self) {
        let started = self.metrics.compaction_started.unwrap_or_else(Instant::now);
        let elapsed = started.elapsed();
        let secs = elapsed.as_secs_f64();
        let took = if secs < 10.0 {
            format!("{secs:.1}s")
        } else {
            format!("{:.0}s", secs)
        };
        let msg = format!("context folded · {took}");
        let painted = self.metrics.compaction_painted;
        if !painted && elapsed < crate::optimistic::LOAD_SHOW_DELAY {
            self.finish_compaction_chrome();
            self.push_fold_card(msg);
            return;
        }
        let min = crate::optimistic::LOAD_MIN_VISIBLE;
        if elapsed < min {
            self.metrics.compaction_hide_at = Some(started + min);
            self.pending_fold = Some(msg);
            self.mark_dirty();
            return;
        }
        self.finish_compaction_chrome();
        self.push_fold_card(msg);
    }

    fn finish_compaction_chrome(&mut self) {
        self.metrics.is_compacting = false;
        self.metrics.compaction_status.clear();
        self.metrics.compaction_started = None;
        self.metrics.compaction_hide_at = None;
        self.metrics.compaction_painted = false;
        self.mark_dirty();
    }

    fn push_fold_card(&mut self, msg: String) {
        self.messages.push(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::Compaction,
            content: msg,
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        self.mark_dirty();
    }

    pub fn release_holds(&mut self) {
        let Some(at) = self.metrics.compaction_hide_at else {
            return;
        };
        if Instant::now() < at {
            return;
        }
        self.finish_compaction_chrome();
        if let Some(msg) = self.pending_fold.take() {
            self.push_fold_card(msg);
        }
    }

    pub fn clamp_modal(&mut self, len: usize) {
        if len == 0 {
            self.modal_selected = 0;
        } else {
            self.modal_selected = self.modal_selected.min(len.saturating_sub(1));
        }
    }

    pub fn filter_matches(hay: &str, q: &str) -> bool {
        q.is_empty() || hay.to_ascii_lowercase().contains(&q.to_ascii_lowercase())
    }

    pub fn filtered_provider_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.providers
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                Self::filter_matches(&p.name, q)
                    || Self::filter_matches(&p.slug, q)
                    || p.models.iter().any(|m| Self::filter_matches(m, q))
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_model_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        let Some(p) = self.providers.get(self.picker_provider) else {
            return Vec::new();
        };
        p.models
            .iter()
            .enumerate()
            .filter(|(_, m)| Self::filter_matches(m, q))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_branch_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.branches
            .iter()
            .enumerate()
            .filter(|(_, b)| Self::filter_matches(&b.name, q))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn apply_session_yolo(&mut self, yolo: bool) {
        if self.metrics.permission_mode == PermissionMode::Plan && !yolo {
            return;
        }
        self.metrics.permission_mode = PermissionMode::from_session_info(yolo);
    }

    pub fn apply_session_plan(&mut self, plan: bool) {
        if plan {
            self.metrics.permission_mode = PermissionMode::Plan;
        } else if self.metrics.permission_mode == PermissionMode::Plan {
            self.metrics.permission_mode = PermissionMode::Manual;
        }
    }

    /// Honor Ink `config.set` chrome keys so the TUI actually changes.
    pub fn apply_config_value(&mut self, key: &str, value: &str) -> bool {
        let v = value.trim();
        match key {
            "density" => {
                self.compact = match v.to_ascii_lowercase().as_str() {
                    "on" | "1" | "true" | "compact" => true,
                    "off" | "0" | "false" | "full" => false,
                    "toggle" => !self.compact,
                    _ => return false,
                };
                true
            }
            "indicator" => {
                let Some(style) = IndicatorStyle::parse(v) else {
                    return false;
                };
                self.indicator = style;
                true
            }
            "statusbar" => {
                let next = match v.to_ascii_lowercase().as_str() {
                    "toggle" => {
                        if self.status_bar == StatusBarMode::Off {
                            StatusBarMode::Top
                        } else {
                            StatusBarMode::Off
                        }
                    }
                    other => match StatusBarMode::parse(other) {
                        Some(m) => m,
                        None => return false,
                    },
                };
                self.status_bar = next;
                true
            }
            "fast" => {
                self.fast_mode = match v.to_ascii_lowercase().as_str() {
                    "fast" | "on" | "1" | "true" | "priority" => true,
                    "normal" | "off" | "0" | "false" => false,
                    "toggle" => !self.fast_mode,
                    _ => return false,
                };
                true
            }
            "busy" => {
                let Some(mode) = BusyMode::parse(v) else {
                    return false;
                };
                self.busy_mode = mode;
                true
            }
            "reasoning" => match v.to_ascii_lowercase().as_str() {
                "show" | "on" => {
                    self.show_thinking = true;
                    true
                }
                "hide" | "off" => {
                    self.show_thinking = false;
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn visible_messages(&self) -> &[ChatMessage] {
        if !self.focus_view {
            return &self.messages;
        }
        let i = self.last_user_index().unwrap_or(0);
        &self.messages[i..]
    }

    pub fn filtered_toolset_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.toolsets
            .iter()
            .enumerate()
            .filter(|(_, t)| {
                Self::filter_matches(&t.name, q) || Self::filter_matches(&t.description, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_spawn_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.spawn_trees
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                Self::filter_matches(&e.label, q)
                    || Self::filter_matches(&e.path, q)
                    || Self::filter_matches(&e.count.to_string(), q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_project_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.projects_list
            .iter()
            .enumerate()
            .filter(|(_, p)| Self::filter_matches(&p.name, q) || Self::filter_matches(&p.id, q))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_project_session_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.project_sessions
            .iter()
            .enumerate()
            .filter(|(_, s)| Self::filter_matches(&s.title, q) || Self::filter_matches(&s.id, q))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_cron_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.cron_jobs
            .iter()
            .enumerate()
            .filter(|(_, j)| {
                Self::filter_matches(&j.name, q)
                    || Self::filter_matches(&j.id, q)
                    || Self::filter_matches(&j.schedule, q)
                    || Self::filter_matches(&j.state, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_plugin_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.plugins
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                Self::filter_matches(&p.name, q) || Self::filter_matches(&p.version, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_mcp_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.mcp_servers
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                Self::filter_matches(&m.name, q)
                    || Self::filter_matches(&m.transport, q)
                    || Self::filter_matches(&m.description, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_palette_entries(&self) -> Vec<&SlashEntry> {
        crate::slash::rank_entries(&self.picker_filter, &self.slash_catalog)
    }

    pub fn slash_ranked(&self) -> Vec<&SlashEntry> {
        if crate::slash::slash_arg_stage(&self.slash_query, self.slash_replace_from) {
            self.slash_gateway.iter().collect()
        } else {
            crate::slash::rank_entries(&self.slash_query, &self.slash_catalog)
        }
    }

    pub fn filtered_skill_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.skills
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                Self::filter_matches(&s.name, q)
                    || Self::filter_matches(&s.category, q)
                    || Self::filter_matches(&s.description, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_profile_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.profiles
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                Self::filter_matches(&p.name, q)
                    || Self::filter_matches(&p.display_name, q)
                    || Self::filter_matches(&p.description, q)
                    || Self::filter_matches(&p.model, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_agent_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.agent_rows
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                Self::filter_matches(&a.title, q)
                    || Self::filter_matches(&a.id, q)
                    || Self::filter_matches(&a.status, q)
                    || Self::filter_matches(&a.last_tool, q)
                    || Self::filter_matches(&a.model, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_memory_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.memory_nodes
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                Self::filter_matches(&m.label, q)
                    || Self::filter_matches(&m.kind, q)
                    || Self::filter_matches(&m.meta, q)
                    || Self::filter_matches(&m.body, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn filtered_session_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.sessions_list
            .iter()
            .enumerate()
            .filter(|(_, s)| Self::filter_matches(&s.title, q) || Self::filter_matches(&s.id, q))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn picker_len(&self) -> usize {
        match self.active_view {
            ActiveView::ModelPicker => match self.picker_stage {
                PickerStage::Providers => self.filtered_provider_indices().len(),
                PickerStage::Models => self.filtered_model_indices().len(),
                PickerStage::Key => 0,
            },
            ActiveView::BranchPicker => self.filtered_branch_indices().len(),
            ActiveView::Skills => self.filtered_skill_indices().len(),
            ActiveView::Sessions => self.filtered_session_indices().len(),
            ActiveView::Profiles => self.filtered_profile_indices().len(),
            ActiveView::Agents => {
                if self.agents_steer {
                    self.agent_rows.len()
                } else {
                    self.filtered_agent_indices().len()
                }
            }
            ActiveView::Memory => self.filtered_memory_indices().len(),
            ActiveView::ThemePicker => crate::palette::THEME_COUNT,
            ActiveView::Tasks => self.tasks.len(),
            ActiveView::Rollback => self.filtered_checkpoint_indices().len(),
            ActiveView::Background => 1 + self.filtered_bg_indices().len(),
            ActiveView::Mcp => self.filtered_mcp_indices().len(),
            ActiveView::Palette => self.filtered_palette_entries().len(),
            ActiveView::Tools => self.filtered_toolset_indices().len(),
            ActiveView::Plugins => self.filtered_plugin_indices().len(),
            ActiveView::Cron => self.filtered_cron_indices().len(),
            ActiveView::Replay => self.filtered_spawn_indices().len(),
            ActiveView::Projects => {
                if self.project_drill.is_some() {
                    self.filtered_project_session_indices().len()
                } else {
                    self.filtered_project_indices().len()
                }
            }
            _ => 0,
        }
    }

    pub fn selected_provider(&self) -> Option<&ModelProvider> {
        self.providers.get(self.picker_provider)
    }

    pub fn clear_picker_key(&mut self) {
        self.picker_key.clear();
        self.picker_key_error.clear();
        self.picker_key_saving = false;
    }

    pub fn open_provider_key(&mut self, provider_index: usize) {
        self.picker_provider = provider_index;
        self.picker_stage = PickerStage::Key;
        self.clear_picker_key();
        self.picker_filter.clear();
        self.mark_dirty();
    }

    pub fn apply_saved_provider(&mut self, provider: ModelProvider) {
        let slug = provider.slug.clone();
        if let Some(i) = self.providers.iter().position(|p| p.slug == slug) {
            self.providers[i] = provider;
            self.picker_provider = i;
        } else {
            self.providers.push(provider);
            self.picker_provider = self.providers.len() - 1;
        }
        self.picker_stage = PickerStage::Models;
        self.clear_picker_key();
        self.picker_filter.clear();
        self.modal_selected = 0;
        let n = self.picker_len();
        self.clamp_modal(n);
        self.mark_dirty();
    }

    pub fn open_picker(&mut self) {
        self.active_view = ActiveView::ModelPicker;
        self.picker_stage = PickerStage::Providers;
        self.picker_filter.clear();
        self.clear_picker_key();
        self.modal_selected = self
            .providers
            .iter()
            .position(|p| p.is_current)
            .unwrap_or(0);
        let n = self.picker_len();
        self.clamp_modal(n);
        self.mark_dirty();
    }

    pub fn open_branch_picker(&mut self) {
        self.active_view = ActiveView::BranchPicker;
        self.picker_filter.clear();
        let current = self.metrics.git_branch.clone();
        self.modal_selected = self
            .branches
            .iter()
            .position(|b| b.current || current.as_deref() == Some(b.name.as_str()))
            .unwrap_or(0);
        let n = self.picker_len();
        self.clamp_modal(n);
        self.mark_dirty();
    }

    pub fn scroll_older(&mut self, lines: usize) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(lines);
        self.mark_dirty();
    }

    pub fn scroll_newer(&mut self, lines: usize) {
        self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(lines);
        self.mark_dirty();
    }

    pub fn scrolled_off_tail(&self) -> bool {
        self.scroll_from_bottom > 0
    }

    pub fn jump_to_tail(&mut self) {
        if self.scroll_from_bottom == 0 {
            return;
        }
        self.scroll_from_bottom = 0;
        self.mark_dirty();
    }

    pub fn start_turn(&mut self, user_text: String) {
        self.mark_session_ready();
        maybe_attach_images(self, &user_text);
        for img in self.pending_images.clone() {
            let path = img.path.to_string_lossy().to_string();
            let already = self
                .messages
                .iter()
                .any(|m| matches!(&m.role, MessageRole::ImagePreview { path: p } if p == &path));
            if !already {
                self.messages.push(ChatMessage {
                    id: uuid::Uuid::new_v4().to_string(),
                    role: MessageRole::ImagePreview { path },
                    content: format!("Image: {}", img.name),
                    timestamp: Local::now(),
                    output: String::new(),
                    is_streaming: false,
                });
            }
        }
        self.pending_images.clear();
        self.prompt_history.push(user_text.clone());
        if self.session_title.is_empty() || self.session_title == "Hermes TUI" {
            let line = user_text.lines().next().unwrap_or("").trim();
            if !line.is_empty() {
                self.session_title = line.chars().take(48).collect();
            }
        }
        self.agent_rows.retain(|r| r.kind != "subagent");
        self.agents_nudged = false;
        self.agents_steer = false;
        self.agents_replay = false;
        self.add_user_message(user_text);
        self.is_generating = true;
        self.metrics.turn_start_time = Some(Instant::now());
        self.metrics.streaming_tokens_count = 0;
        self.metrics.tokens_per_sec = 0.0;
        self.metrics.activity = "thinking".into();
        self.scroll_from_bottom = 0;
        self.trace_follow = true;
        self.trim_old_messages();
        self.mark_dirty();
    }

    pub fn add_user_message(&mut self, text: String) {
        self.messages.push(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::User,
            content: text,
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        self.mark_dirty();
    }

    pub fn add_system(&mut self, text: impl Into<String>) {
        self.messages.push(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::System,
            content: text.into(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: false,
        });
        self.mark_dirty();
    }

    pub fn append_assistant_delta(&mut self, delta: &str) {
        // Rough byte-to-token heuristic for a live speedometer only.
        self.metrics.streaming_tokens_count += (delta.len() as u64 / 4).max(1);
        if let Some(start) = self.metrics.turn_start_time {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 0.2 {
                self.metrics.tokens_per_sec = self.metrics.streaming_tokens_count as f64 / elapsed;
            }
        }

        if let Some(idx) = self.last_turn_reasoning_index() {
            self.messages[idx].is_streaming = false;
        }

        if let Some(idx) = self.turn_assistant_index() {
            self.messages[idx].content.push_str(delta);
            self.messages[idx].is_streaming = true;
            self.mark_dirty();
            return;
        }
        self.messages.push(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::Assistant,
            content: delta.to_string(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: true,
        });
        self.trim_old_messages();
        self.mark_dirty();
    }

    pub fn last_user_text(&self) -> Option<String> {
        self.last_user_index()
            .and_then(|i| self.messages.get(i))
            .map(|m| m.content.clone())
            .filter(|s| !s.trim().is_empty())
    }

    pub fn toggle_details(&mut self) {
        let ids: Vec<String> = self
            .messages
            .iter()
            .filter(|m| matches!(m.role, MessageRole::Tool { .. } | MessageRole::Reasoning))
            .map(|m| m.id.clone())
            .collect();
        if ids.is_empty() {
            self.set_toast("no tool or thought cards");
            return;
        }
        let all_open = ids.iter().all(|id| self.expanded_tools.contains(id));
        if all_open {
            self.expanded_tools.clear();
            self.set_toast("details collapsed");
        } else {
            self.expanded_tools.extend(ids);
            self.set_toast("details expanded");
        }
        self.expand_epoch = self.expand_epoch.wrapping_add(1);
        self.mark_dirty();
    }

    pub fn last_user_index(&self) -> Option<usize> {
        self.messages
            .iter()
            .rposition(|m| m.role == MessageRole::User)
    }

    /// Latest thought in this turn. Consecutive deltas join this row;
    /// a new row starts after a tool or assistant message.
    fn last_turn_reasoning_index(&self) -> Option<usize> {
        let start = self.last_user_index().unwrap_or(0);
        self.messages[start..]
            .iter()
            .rposition(|m| m.role == MessageRole::Reasoning)
            .map(|i| start + i)
    }

    fn turn_assistant_index(&self) -> Option<usize> {
        let start = self.last_user_index().unwrap_or(0);
        self.messages[start..]
            .iter()
            .rposition(|m| m.role == MessageRole::Assistant)
            .map(|i| start + i)
    }

    fn trim_old_messages(&mut self) {
        const MAX: usize = 400;
        if self.messages.len() <= MAX {
            return;
        }
        let keep_from = self.last_user_index().unwrap_or(0);
        let extra = self.messages.len() - MAX;
        let drop = extra.min(keep_from);
        if drop > 0 {
            self.messages.drain(0..drop);
        }
    }

    pub fn append_reasoning_delta(&mut self, delta: &str) {
        if delta.is_empty() || is_thinking_status(delta) {
            return;
        }
        let at_tail = matches!(
            self.messages.last().map(|m| &m.role),
            Some(MessageRole::Reasoning)
        );
        if at_tail {
            if let Some(idx) = self.last_turn_reasoning_index() {
                let msg = &mut self.messages[idx];
                if msg.content.is_empty() {
                    msg.content.push_str(delta);
                } else if delta.starts_with(msg.content.as_str()) && delta.len() > msg.content.len()
                {
                    // Gateway sometimes sends a growing snapshot, not a tail delta.
                    msg.content = delta.to_string();
                } else if !msg.content.ends_with(delta) {
                    msg.content.push_str(delta);
                }
                msg.is_streaming = true;
                self.mark_dirty();
                return;
            }
        }
        self.messages.push(ChatMessage {
            id: uuid::Uuid::new_v4().to_string(),
            role: MessageRole::Reasoning,
            content: delta.to_string(),
            timestamp: Local::now(),
            output: String::new(),
            is_streaming: true,
        });
        self.mark_dirty();
    }

    /// Fail-closed when the event bus skipped frames. Never leave a stuck turn bar.
    pub fn note_lagged(&mut self, skipped: u64) {
        if skipped == 0 {
            return;
        }
        if self.is_generating {
            self.finish_streaming();
        }
        self.set_toast(format!(
            "dropped {skipped} events · stream may be stale · send again"
        ));
    }

    pub fn trim_last_user_turn(&mut self) {
        if let Some(i) = self.last_user_index() {
            self.messages.truncate(i);
            self.mark_dirty();
        }
    }

    pub fn filtered_checkpoint_indices(&self) -> Vec<usize> {
        let q = self.picker_filter.as_str();
        self.checkpoints
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                Self::filter_matches(&c.hash, q)
                    || Self::filter_matches(&c.message, q)
                    || Self::filter_matches(&c.timestamp, q)
            })
            .map(|(i, _)| i)
            .collect()
    }

    pub fn finish_streaming(&mut self) {
        self.messages.retain(|m| {
            if m.role != MessageRole::Reasoning {
                return true;
            }
            let t = m.content.trim();
            !t.is_empty() && !is_thinking_status(t)
        });
        for msg in &mut self.messages {
            msg.is_streaming = false;
        }
        self.is_generating = false;
        self.trace_focus = false;
        self.metrics.active_tool = None;
        self.metrics.turn_start_time = None;
        self.metrics.tokens_per_sec = 0.0;
        self.metrics.activity.clear();
        self.mark_dirty();
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.metrics
            .turn_start_time
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0)
    }

    pub fn append_running_tool_output(
        &mut self,
        name: Option<&str>,
        tool_id: Option<&str>,
        chunk: &str,
    ) {
        let chunk = chunk.trim_end_matches('\0');
        if chunk.is_empty() {
            return;
        }
        if let Some(msg) = self.messages.iter_mut().rev().find(|m| match &m.role {
            MessageRole::Tool {
                name: n,
                status,
                tool_id: id,
            } if status.contains("running") => {
                let id_ok = match (tool_id, id.as_deref()) {
                    (Some(want), Some(have)) => want == have,
                    (Some(_), None) => false,
                    (None, _) => true,
                };
                let name_ok = name.map(|w| w == n).unwrap_or(true);
                id_ok && name_ok
            }
            _ => false,
        }) {
            if !msg.output.is_empty() && !msg.output.ends_with('\n') && !chunk.starts_with('\n') {
                msg.output.push('\n');
            }
            msg.output.push_str(chunk);
            const CAP: usize = 32 * 1024;
            if msg.output.len() > CAP {
                let drain = msg.output.len() - CAP;
                msg.output.drain(..drain);
            }
            self.mark_dirty();
        }
    }

    pub fn turn_detail(&self) -> Option<String> {
        if let Some(msg) = self.last_running_tool() {
            if let Some(line) = msg
                .output
                .lines()
                .rev()
                .map(crate::ui::stream::strip_ansi)
                .find(|l| !l.trim().is_empty())
            {
                let line = line.trim().to_string();
                let label = self.live_status();
                if !line.is_empty() && !label.contains(line.trim()) {
                    return Some(crate::tips::ellipsize(&line, 72));
                }
            }
        }
        let act = self.metrics.activity.trim();
        if act.is_empty() || is_wait_activity(act) {
            return None;
        }
        let label = self.live_status().to_ascii_lowercase();
        let low = act.to_ascii_lowercase();
        if label.contains(&low) {
            None
        } else {
            Some(crate::tips::ellipsize(act, 72))
        }
    }

    pub fn complete_tool(
        &mut self,
        name: Option<&str>,
        tool_id: Option<&str>,
        error: bool,
        duration_s: Option<f64>,
    ) {
        let dur = duration_s
            .filter(|s| *s >= 0.05)
            .map(crate::ui::turn_bar::fmt_duration);
        let status = match (error, dur) {
            (true, Some(d)) => format!("failed · {d}"),
            (true, None) => "failed".into(),
            (false, Some(d)) => format!("completed · {d}"),
            (false, None) => "completed".into(),
        };
        for msg in self.messages.iter_mut().rev() {
            if let MessageRole::Tool {
                name: n,
                status: st,
                tool_id: id,
            } = &mut msg.role
            {
                let id_ok = match (tool_id, id.as_deref()) {
                    (Some(want), Some(have)) => want == have,
                    (Some(_), None) => false,
                    (None, _) => true,
                };
                let name_ok = name.map(|w| w == n).unwrap_or(true);
                if st.contains("running") && id_ok && name_ok {
                    *st = status.to_string();
                    break;
                }
            }
        }
        let todo_tool = name.is_some_and(|n| n.to_ascii_lowercase().contains("todo"));
        if !todo_tool {
            if let Some(last_task) = self
                .tasks
                .iter_mut()
                .rev()
                .find(|t| t.status == TaskStatus::InProgress)
            {
                last_task.status = if error {
                    TaskStatus::Failed
                } else {
                    TaskStatus::Completed
                };
            }
        }
        self.metrics.active_tool = None;
        if self.split_diff {
            self.refresh_diff();
        }
        self.mark_dirty();
    }

    pub fn assistant_text(&self, n: Option<usize>) -> Option<&str> {
        let asst: Vec<&str> = self
            .messages
            .iter()
            .filter_map(|m| match &m.role {
                MessageRole::Assistant if !m.content.is_empty() => Some(m.content.as_str()),
                _ => None,
            })
            .collect();
        if asst.is_empty() {
            return None;
        }
        let idx = match n {
            None | Some(0) => asst.len() - 1,
            Some(i) => i.min(asst.len()).saturating_sub(1),
        };
        Some(asst[idx])
    }

    pub fn copy_latest_response(&mut self) -> bool {
        self.copy_assistant(None)
    }

    pub fn copy_assistant(&mut self, n: Option<usize>) -> bool {
        if n.is_none() {
            if let Some(path) = self.messages.iter().rev().find_map(|m| match &m.role {
                MessageRole::ImagePreview { path } => Some(path.clone()),
                _ => None,
            }) {
                if self
                    .messages
                    .iter()
                    .rev()
                    .find(|m| matches!(m.role, MessageRole::Assistant if !m.content.is_empty()))
                    .is_none()
                {
                    match crate::platform::copy_to_clipboard(&path) {
                        Ok(()) => self.set_toast("Copied image path"),
                        Err(e) => self.set_toast(format!("Copy failed: {e}")),
                    }
                    return true;
                }
            }
        }
        let Some(text) = self.assistant_text(n).map(str::to_string) else {
            return false;
        };
        match crate::platform::copy_to_clipboard(&text) {
            Ok(()) => self.set_toast("Copied assistant reply"),
            Err(e) => self.set_toast(format!("Copy failed: {e}")),
        }
        true
    }

    pub fn open_latest_media(&mut self) -> bool {
        for msg in self.messages.iter().rev() {
            if let MessageRole::ImagePreview { path } = &msg.role {
                match crate::platform::open_path(path) {
                    Ok(()) => self.set_toast(format!("Opened {path}")),
                    Err(e) => self.set_toast(format!("Open failed: {e}")),
                }
                return true;
            }
        }
        match crate::platform::open_path(&self.metrics.cwd) {
            Ok(()) => self.set_toast(format!("Opened {}", self.metrics.cwd)),
            Err(e) => self.set_toast(format!("Open failed: {e}")),
        }
        true
    }
}
