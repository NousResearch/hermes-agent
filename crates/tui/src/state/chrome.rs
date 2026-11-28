//! Files, work, diff, and trace chrome on `AppState`.
use super::*;

impl AppState {
    pub fn new() -> Self {
        Self {
            session_id: None,
            vim: None,
            session_key: String::new(),
            pending_memory_edit: None,
            pending_memory_body: String::new(),
            mcp_key_name: None,
            startup_resume: None,
            bg_tasks: Vec::new(),
            session_title: "Hermes TUI".into(),
            session_started: Instant::now(),
            messages: Vec::new(),
            metrics: SessionMetrics::default(),
            is_generating: false,
            show_thinking: false,
            active_view: ActiveView::Chat,
            modal_selected: 0,
            goal: None,
            tasks: Vec::new(),
            sessions_list: Vec::new(),
            skills: Vec::new(),
            profiles: Vec::new(),
            agent_rows: Vec::new(),
            agents_paused: false,
            agents_caps: String::new(),
            agents_steer: false,
            agents_replay: false,
            agents_nudged: false,
            spawn_trees: Vec::new(),
            projects_list: Vec::new(),
            project_sessions: Vec::new(),
            project_drill: None,
            compact: false,
            indicator: IndicatorStyle::Unicode,
            status_bar: StatusBarMode::Bottom,
            fast_mode: false,
            busy_mode: BusyMode::Queue,
            memory_summary: Vec::new(),
            memory_nodes: Vec::new(),
            providers: Vec::new(),
            picker_stage: PickerStage::Providers,
            picker_provider: 0,
            picker_list: None,
            picker_offset: 0,
            picker_filter: String::new(),
            picker_key: String::new(),
            picker_key_error: String::new(),
            picker_key_saving: false,
            hit_model: None,
            hit_branch: None,
            hit_mode: None,
            hit_context: None,
            hit_session: None,
            hit_bg: None,
            hit_agents: None,
            hit_process: None,
            hit_dock: Vec::new(),
            hit_dock_stop: Vec::new(),
            hit_dock_bar: None,
            hover: HoverKind::None,
            click_flash: None,
            composer_area: None,
            queue_area: None,
            hit_queue: Vec::new(),
            stream_area: None,
            hit_tools: Vec::new(),
            expanded_tools: HashSet::new(),
            expand_epoch: 0,
            branches: Vec::new(),
            slash_open: false,
            slash_query: String::new(),
            slash_selected: 0,
            slash_catalog: local_entries(),
            slash_gateway: Vec::new(),
            slash_replace_from: 1,
            complete_open: false,
            complete_items: Vec::new(),
            complete_selected: 0,
            complete_replace_from: 0,
            pending_images: Vec::new(),
            paste_chips: Vec::new(),
            peek_title: String::new(),
            peek_body: String::new(),
            peek_image: None,
            peek_offset: 0,
            hit_pastes: Vec::new(),
            checkpoints: Vec::new(),
            checkpoints_enabled: true,
            rollback_diff: String::new(),
            dirty: true,
            scroll_from_bottom: 0,
            prompt_history: PromptHistory::default(),
            pending_approval: None,
            pending_clarify: None,
            pending_secret: None,
            protocol_warned: false,
            prompt_queue: VecDeque::new(),
            queue_edit: None,
            split_trace: false,
            split_diff: false,
            split_files: false,
            split_work: false,
            work_focus: false,
            work_selected: 0,
            work_offset: 0,
            work_list: None,
            work_show_diff: false,
            work_dirty: String::new(),
            work_diff_files: Vec::new(),
            work_diff_selected: 0,
            work_diff_offset: 0,
            files_focus: false,
            files_rows: Vec::new(),
            files_selected: 0,
            files_offset: 0,
            files_expanded: HashSet::new(),
            files_status: HashMap::new(),
            files_preview: String::new(),
            files_list: None,
            diff_text: String::new(),
            diff_tool_id: None,
            theme_id: "gold".into(),
            theme_revert: None,
            hermes_home: crate::config::default_hermes_home(),
            trace_open: false,
            trace_focus: false,
            trace_follow: true,
            trace_selected: 0,
            resume_step: None,
            intro_tools: Vec::new(),
            intro_skills: Vec::new(),
            mcp_connected: 0,
            mcp_servers: Vec::new(),
            toolsets: Vec::new(),
            plugins: Vec::new(),
            cron_jobs: Vec::new(),
            shell_context: String::new(),
            focus_view: false,
            want_attention: false,
            mouse_on: true,
            release_date: String::new(),
            intro_warning: None,
            session_ready: false,
            reveal_started: None,
            tips_open: true,
            tip_index: crate::tips::start_index(),
            tip_shown_at: Instant::now(),
            hit_tips_close: None,
            hit_tips_bar: None,
            hit_jump: None,
            yolo_epoch: 0,
            model_epoch: 0,
            pending_undo: None,
            pending_fold: None,
            armed: None,
        }
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn tool_steps(&self) -> Vec<ToolStep> {
        self.messages
            .iter()
            .enumerate()
            .filter_map(|(msg_index, m)| {
                let MessageRole::Tool {
                    name,
                    status,
                    tool_id,
                } = &m.role
                else {
                    return None;
                };
                Some(ToolStep {
                    index: 0,
                    name: name.clone(),
                    status: status.clone(),
                    args: m.content.clone(),
                    tool_id: tool_id.clone(),
                    msg_index,
                })
            })
            .enumerate()
            .map(|(i, mut step)| {
                step.index = i + 1;
                step
            })
            .collect()
    }

    pub fn selected_step(&self) -> Option<ToolStep> {
        let steps = self.tool_steps();
        if steps.is_empty() {
            return None;
        }
        let i = self.trace_selected.min(steps.len() - 1);
        steps.into_iter().nth(i)
    }

    pub fn refresh_diff(&mut self) {
        if self.diff_tool_id.is_some() {
            return;
        }
        self.diff_text = crate::platform::git_diff_snapshot(&self.metrics.cwd);
    }

    pub fn toggle_edit_diff(&mut self, id: &str) {
        if self.split_diff && self.diff_tool_id.as_deref() == Some(id) {
            self.split_diff = false;
            self.diff_tool_id = None;
            self.expand_epoch = self.expand_epoch.wrapping_add(1);
            self.mark_dirty();
            return;
        }
        self.split_diff = true;
        self.split_work = false;
        self.split_files = false;
        self.work_focus = false;
        self.files_focus = false;
        self.diff_tool_id = Some(id.to_string());
        self.diff_text = self
            .edit_diff_for(id)
            .unwrap_or_else(|| "(no diff for this edit)".into());
        self.expand_epoch = self.expand_epoch.wrapping_add(1);
        self.mark_dirty();
    }

    pub fn close_edit_diff(&mut self) {
        if self.diff_tool_id.take().is_some() {
            self.split_diff = false;
            self.expand_epoch = self.expand_epoch.wrapping_add(1);
            self.mark_dirty();
        }
    }

    fn edit_diff_for(&self, id: &str) -> Option<String> {
        let msg = self.messages.iter().find(|m| m.id == id)?;
        let patch = crate::ui::stream::edit_patch(&msg.content, &msg.output);
        if patch.trim().is_empty() {
            None
        } else {
            Some(patch)
        }
    }

    pub fn refresh_files(&mut self) {
        let cwd = std::path::PathBuf::from(&self.metrics.cwd);
        self.files_status = crate::fs_tree::parse_porcelain(
            &crate::platform::git_status_porcelain(&self.metrics.cwd),
        );
        let keep = self
            .files_rows
            .get(self.files_selected)
            .map(|r| r.rel.clone());
        self.files_rows =
            crate::fs_tree::visible_rows(&cwd, &self.files_expanded, &self.files_status);
        if let Some(rel) = keep {
            if let Some(i) = self.files_rows.iter().position(|r| r.rel == rel) {
                self.files_selected = i;
            }
        }
        if self.files_selected >= self.files_rows.len() {
            self.files_selected = self.files_rows.len().saturating_sub(1);
        }
        self.load_file_preview();
        self.mark_dirty();
    }

    pub fn load_file_preview(&mut self) {
        let Some(row) = self.files_rows.get(self.files_selected) else {
            self.files_preview.clear();
            return;
        };
        if row.is_dir {
            self.files_preview = format!("  {}/\n  enter expand  o open", row.rel);
            return;
        }
        self.files_preview = crate::platform::git_file_diff(&self.metrics.cwd, &row.rel);
    }

    pub fn files_move(&mut self, delta: i32) {
        let n = self.files_rows.len() as i32;
        if n == 0 {
            return;
        }
        let next = (self.files_selected as i32 + delta).clamp(0, n - 1) as usize;
        if next != self.files_selected {
            self.files_selected = next;
            self.load_file_preview();
            self.mark_dirty();
        }
    }

    pub fn files_activate(&mut self) {
        let Some(row) = self.files_rows.get(self.files_selected).cloned() else {
            return;
        };
        if row.is_dir {
            if !self.files_expanded.remove(&row.rel) {
                self.files_expanded.insert(row.rel);
            }
            self.refresh_files();
        } else {
            self.load_file_preview();
            self.mark_dirty();
        }
    }

    pub fn files_open_selected(&self) {
        let Some(row) = self.files_rows.get(self.files_selected) else {
            return;
        };
        let path = std::path::Path::new(&self.metrics.cwd).join(&row.rel);
        let _ = crate::platform::open_path(&path.to_string_lossy());
    }

    pub fn files_restore_selected(&mut self) -> String {
        let Some(row) = self.files_rows.get(self.files_selected).cloned() else {
            return "no file selected".into();
        };
        if row.is_dir {
            return "pick a file to restore".into();
        }
        let path = std::path::Path::new(&self.metrics.cwd).join(&row.rel);
        let undo = if path.exists() {
            match crate::platform::read_worktree_bytes(&self.metrics.cwd, &row.rel) {
                Ok(b) if b.len() <= crate::optimistic::UNDO_MAX_BYTES => Some(PendingUndo {
                    kind: UndoKind::File {
                        rel: row.rel.clone(),
                        previous: Some(b),
                    },
                    created: Instant::now(),
                }),
                _ => None,
            }
        } else {
            Some(PendingUndo {
                kind: UndoKind::File {
                    rel: row.rel.clone(),
                    previous: None,
                },
                created: Instant::now(),
            })
        };
        match crate::platform::git_restore_worktree(&self.metrics.cwd, &row.rel) {
            Ok(()) => {
                self.pending_undo = undo;
                self.refresh_files();
                if self.pending_undo.is_some() {
                    format!("restored {} · u undo", row.rel)
                } else {
                    format!("restored {}", row.rel)
                }
            }
            Err(e) => format!("restore failed: {e}"),
        }
    }

    pub fn want_diff(&self, cols: u16) -> bool {
        self.split_diff && !self.split_files && !self.split_work && cols >= 72
    }

    pub fn want_files(&self, cols: u16) -> bool {
        self.split_files && cols >= 70
    }

    pub fn want_work(&self, cols: u16) -> bool {
        self.split_work && !self.split_files && cols >= 70
    }

    pub fn toggle_work(&mut self) {
        self.split_work = !self.split_work;
        if self.split_work {
            self.work_focus = true;
            self.files_focus = false;
            self.split_files = false;
            self.trace_focus = false;
            self.refresh_work_chrome();
        } else {
            self.work_focus = false;
            self.work_list = None;
        }
        self.mark_dirty();
    }

    pub fn refresh_work_chrome(&mut self) {
        let cwd = &self.metrics.cwd;
        self.work_dirty = crate::platform::git_dirty_summary(cwd);
        if self.work_show_diff {
            let porcelain = crate::platform::git_status_porcelain(cwd);
            let timeout = std::time::Duration::from_millis(1500);
            let check = crate::platform::git_stdout(cwd, &["diff", "--check", "HEAD"], timeout)
                .or_else(|| crate::platform::git_stdout(cwd, &["diff", "--check"], timeout))
                .unwrap_or_default();
            self.work_diff_files = crate::platform::list_dirty_files(&porcelain, &check);
            if self.work_diff_files.is_empty() {
                self.work_diff_selected = 0;
                self.diff_text = crate::platform::git_diff_check(cwd);
            } else {
                self.work_diff_selected =
                    self.work_diff_selected.min(self.work_diff_files.len() - 1);
                self.load_work_diff_preview();
            }
            self.work_diff_offset = 0;
        }
        let n = self.agent_rows.len();
        if n == 0 {
            self.work_selected = 0;
        } else {
            self.work_selected = self.work_selected.min(n - 1);
        }
        self.mark_dirty();
    }

    pub fn load_work_diff_preview(&mut self) {
        let Some(file) = self.work_diff_files.get(self.work_diff_selected) else {
            self.diff_text.clear();
            return;
        };
        let rel = file.rel.clone();
        let check = file.check.clone();
        let mut out = String::new();
        if !check.is_empty() {
            out.push_str("## diff --check\n");
            for note in &check {
                out.push_str(note);
                out.push('\n');
            }
            out.push('\n');
        }
        out.push_str("## ");
        out.push_str(&rel);
        out.push('\n');
        out.push_str(&crate::platform::git_file_patch(&self.metrics.cwd, &rel));
        self.diff_text = out;
        self.work_diff_offset = 0;
        self.mark_dirty();
    }

    pub fn work_move(&mut self, delta: i32) {
        if self.work_show_diff {
            let n = self.work_diff_files.len() as i32;
            if n == 0 {
                self.work_diff_selected = 0;
                return;
            }
            let next = (self.work_diff_selected as i32 + delta).clamp(0, n - 1) as usize;
            if next != self.work_diff_selected {
                self.work_diff_selected = next;
                self.load_work_diff_preview();
            }
            return;
        }
        let n = self.agent_rows.len() as i32;
        if n == 0 {
            self.work_selected = 0;
            return;
        }
        let next = (self.work_selected as i32 + delta).clamp(0, n - 1) as usize;
        if next != self.work_selected {
            self.work_selected = next;
            self.mark_dirty();
        }
    }

    pub fn work_scroll_diff(&mut self, delta: i32) {
        if !self.work_show_diff {
            return;
        }
        let max = self.diff_text.lines().count().saturating_sub(1);
        let next = (self.work_diff_offset as i32 + delta).clamp(0, max as i32) as usize;
        if next != self.work_diff_offset {
            self.work_diff_offset = next;
            self.mark_dirty();
        }
    }

    pub fn work_selected_row(&self) -> Option<&AgentRow> {
        self.agent_rows.get(self.work_selected)
    }

    pub fn want_trace(&self, cols: u16) -> bool {
        if !self.split_trace && !self.trace_focus {
            return false;
        }
        cols >= 64
    }

    pub fn follow_latest_tool(&mut self) {
        if !self.trace_follow {
            return;
        }
        let n = self
            .messages
            .iter()
            .filter(|m| matches!(m.role, MessageRole::Tool { .. }))
            .count();
        if n > 0 {
            self.trace_selected = n - 1;
        }
    }
}
