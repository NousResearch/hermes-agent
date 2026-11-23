//! Native port of official Hermes v0.23.3 tools/memory_tool{,_store}.py.
//!
//! A profile owns MEMORY.md and USER.md. System-prompt snapshots are frozen at
//! session open; approved writes update disk, never that session's prefix.
//! Source: NousResearch/hermes-agent caafe5389ae585ef77387a962aee05b1bcf17029.

mod persistence;
mod scan;

use anyhow::{Result, ensure};
use serde_json::{Map, Value, json};
use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

pub const ENTRY_DELIMITER: &str = "\n§\n";
pub const DEFAULT_MEMORY_CHAR_LIMIT: usize = 2200;
pub const DEFAULT_USER_CHAR_LIMIT: usize = 1375;
const PATCH_DROP_HINT: &str = "To drop text inside an entry, patch a longer span that contains it; 'remove' deletes the whole entry.";
const REPLACE_REMOVED: &str = "'replace' is not available; use 'patch'. old_text is the exact span to change and content is its new text; the rest of the entry is kept. To rewrite a whole entry, set old_text to the whole entry. Nothing was written.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryConfig {
    pub memory_char_limit: usize,
    pub user_char_limit: usize,
    pub memory_enabled: bool,
    pub user_profile_enabled: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_char_limit: DEFAULT_MEMORY_CHAR_LIMIT,
            user_char_limit: DEFAULT_USER_CHAR_LIMIT,
            memory_enabled: true,
            user_profile_enabled: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Memory,
    User,
}
impl Target {
    fn name(self) -> &'static str {
        match self {
            Self::Memory => "memory",
            Self::User => "user",
        }
    }
    fn filename(self) -> &'static str {
        match self {
            Self::Memory => "MEMORY.md",
            Self::User => "USER.md",
        }
    }
    fn index(self) -> usize {
        usize::from(self == Self::User)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Action {
    Add,
    Patch,
    Remove,
}
impl Action {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Patch => "patch",
            Self::Remove => "remove",
        }
    }
}

#[derive(Clone)]
struct Operation {
    action: Action,
    content: String,
    old_text: String,
    matched_entry: Option<String>,
}

/// A dry-run validated, profile-bound write. Only this type may cross the
/// approval boundary: destructive operations pin the exact reviewed entry.
#[derive(Clone)]
pub struct PreparedWrite {
    dir: PathBuf,
    target: Target,
    operations: Vec<Operation>,
    single: bool,
    review: Value,
}
impl PreparedWrite {
    pub fn review_payload(&self) -> &Value {
        &self.review
    }
}

pub struct MemoryStore {
    dir: PathBuf,
    config: MemoryConfig,
    snapshot: [String; 2],
    warnings: Vec<String>,
    consolidation_failures: usize,
}

impl MemoryStore {
    /// `memories_dir` is the explicit profile's private memories directory.
    /// It is never derived from a project or from a global mutable environment.
    pub fn open(memories_dir: &Path) -> Result<Self> {
        Self::with_config(memories_dir, MemoryConfig::default())
    }

    pub fn with_config(memories_dir: &Path, config: MemoryConfig) -> Result<Self> {
        let dir = persistence::private_dir(memories_dir)?;
        let mut store = Self {
            dir,
            config,
            snapshot: Default::default(),
            warnings: Vec::new(),
            consolidation_failures: 0,
        };
        for target in [Target::Memory, Target::User] {
            if !store.enabled(target) {
                continue;
            }
            match store.read(target) {
                Ok(raw) => {
                    let entries = parse_entries(&raw);
                    let sanitized = entries.iter().map(|entry| {
                        if entry.starts_with("[BLOCKED:") { return entry.clone(); }
                        let findings = scan::findings(entry);
                        if findings.is_empty() { entry.clone() } else {
                            format!("[BLOCKED: {} entry contained threat pattern(s): {}. Removed from system prompt; use memory(action=remove, target='{}') to delete the original.]", target.filename(), findings.join(", "), target.name())
                        }
                    }).collect::<Vec<_>>();
                    store.snapshot[target.index()] = store.render_block(target, &sanitized);
                    if char_count(&entries) > store.limit(target) {
                        store.warnings.push(format!(
                            "{} exceeds its char limit; existing values were preserved",
                            target.filename()
                        ));
                    }
                }
                Err(error) => store.warnings.push(format!(
                    "Could not load {}: {error}; writes will refuse unreadable content",
                    target.filename()
                )),
            }
        }
        Ok(store)
    }

    /// The exact public tool schema, narrowed when one target is disabled.
    pub fn tool_spec(&self) -> Option<Value> {
        let targets = [Target::Memory, Target::User]
            .into_iter()
            .filter(|target| self.enabled(*target))
            .collect::<Vec<_>>();
        if targets.is_empty() {
            return None;
        }
        let mut schema: Value = serde_json::from_str(include_str!("memory/schema.json"))
            .expect("official static memory schema");
        schema["parameters"]["properties"]["target"]["enum"] = json!(
            targets
                .iter()
                .map(|target| target.name())
                .collect::<Vec<_>>()
        );
        if targets.len() == 1 {
            let (description, replacement) = if targets[0] == Target::Memory {
                (
                    "The enabled built-in store: 'memory' for personal notes.",
                    "TARGET: only 'memory' is enabled for personal notes (environment, conventions, tool quirks, lessons).",
                )
            } else {
                (
                    "The enabled built-in store: 'user' for user profile.",
                    "TARGET: only 'user' is enabled for user profile facts (name, role, preferences, style).",
                )
            };
            schema["parameters"]["properties"]["target"]["description"] = json!(description);
            let full = schema["description"].as_str().unwrap_or_default().replace("TARGETS: 'user' = who the user is (name, role, preferences, style). 'memory' = your notes (environment, conventions, tool quirks, lessons).", replacement);
            schema["description"] = json!(full);
        }
        Some(json!({"type":"function","function":schema}))
    }

    pub fn reset_turn(&mut self) {
        self.consolidation_failures = 0;
    }
    pub fn frozen_context(&self) -> Vec<String> {
        self.snapshot
            .iter()
            .filter(|block| !block.is_empty())
            .cloned()
            .collect()
    }
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    pub fn entries(&self, target: &str) -> Result<Vec<String>> {
        let target = parse_target(Some(&json!(target))).map_err(tool_error)?;
        let _lock = persistence::lock(&self.dir, target.filename())?;
        Ok(parse_entries(&self.read(target)?))
    }

    /// Validate and pin against a fresh locked read, without changing files.
    /// Display review_payload in the approval UI, then consume this value with
    /// apply_approved. Never deserialize a model-provided matched_entry as a pin.
    pub fn prepare(&mut self, args: &Value) -> Result<PreparedWrite> {
        let result = self.prepare_inner(args);
        match result {
            Ok(write) => Ok(write),
            Err(error) => Err(self.count_failure(error)),
        }
    }

    fn prepare_inner(&self, args: &Value) -> Result<PreparedWrite> {
        ensure!(args.is_object(), "Memory arguments must be a JSON object");
        let target = parse_target(args.get("target")).map_err(tool_error)?;
        self.ensure_enabled(target)?;
        let _lock = persistence::lock(&self.dir, target.filename())?;
        let before = parse_entries(&self.read(target)?);
        let (mut operations, single) = parse_operations(args, target).map_err(|mut failure| {
            if !args
                .get("operations")
                .and_then(Value::as_array)
                .is_some_and(|ops| !ops.is_empty())
                && matches!(
                    args.get("action").and_then(Value::as_str),
                    Some("patch" | "remove")
                )
                && args
                    .get("old_text")
                    .and_then(Value::as_str)
                    .is_none_or(str::is_empty)
            {
                failure.value["current_entries"] = json!(before);
                failure.value["usage"] = json!(self.usage(target, &before));
            }
            tool_error(failure)
        })?;
        let planned = self
            .plan(target, &before, &mut operations, true, single)
            .map_err(tool_error)?;
        let _ = planned;
        let review = json!({"target":target.name(),"operations":operations.iter().map(|op| {
            let mut value = json!({"action":op.action.name()});
            if op.action != Action::Remove { value["content"] = json!(op.content); }
            if op.action != Action::Add { value["old_text"] = json!(op.old_text); value["matched_entry"] = json!(op.matched_entry); }
            value
        }).collect::<Vec<_>>()});
        Ok(PreparedWrite {
            dir: self.dir.clone(),
            target,
            operations,
            single,
            review,
        })
    }

    /// Called only after the user approves this exact PreparedWrite.
    /// The final file is atomically replaced after all operations validate.
    pub fn apply_approved(&mut self, mut write: PreparedWrite) -> Result<Value> {
        ensure!(
            write.dir == self.dir,
            "Approved memory write belongs to a different profile"
        );
        self.ensure_enabled(write.target)?;
        let target = write.target;
        let _lock = persistence::lock(&self.dir, target.filename())?;
        let raw = self.read(target)?;
        let before = parse_entries(&raw);
        let plan = match self.plan(target, &before, &mut write.operations, false, write.single) {
            Ok(plan) => plan,
            Err(failure) => return Ok(self.failure_result(failure)),
        };
        let oversized = before
            .iter()
            .any(|entry| entry.chars().count() > self.limit(target));
        let skip_drift = write.single && write.operations[0].action == Action::Add;
        let mut extra = plan.extra;
        if oversized && !skip_drift {
            let only_deletes = plan.entries.iter().all(|entry| before.contains(entry));
            if !only_deletes {
                let backup = persistence::backup(&self.dir, target.filename(), &raw)?;
                return Ok(
                    json!({"success":false,"error":format!("Refusing to write {}: it holds an entry larger than the whole store's char limit. Rewriting it would lose free-form text. Resolve the drift first; nothing was written.", target.filename()),"drift_backup":backup}),
                );
            }
            if before.iter().any(|entry| {
                entry.chars().count() > self.limit(target) && !plan.entries.contains(entry)
            }) {
                extra.insert(
                    "drift_backup".into(),
                    json!(persistence::backup(&self.dir, target.filename(), &raw)?),
                );
            }
        }
        // A duplicate add is a true no-op, including for a store over a lowered cap.
        if plan.entries != before {
            persistence::write(
                &self.dir,
                target.filename(),
                &plan.entries.join(ENTRY_DELIMITER),
            )?;
        }
        self.consolidation_failures = 0;
        let message = if write.single {
            match write.operations[0].action {
                Action::Add if plan.entries == before => {
                    "Entry already exists (no duplicate added).".to_owned()
                }
                Action::Add => "Entry added.".to_owned(),
                Action::Patch => "Entry patched.".to_owned(),
                Action::Remove => "Entry removed.".to_owned(),
            }
        } else {
            format!("Applied {} operation(s).", write.operations.len())
        };
        let mut response = json!({"success":true,"done":true,"target":target.name(),"usage":self.usage_pct(target,char_count(&plan.entries)),"entry_count":plan.entries.len(),"message":message,"note":"Write saved. This update is complete — do not repeat it."});
        response
            .as_object_mut()
            .expect("response object")
            .extend(extra);
        Ok(response)
    }

    fn plan(
        &self,
        target: Target,
        before: &[String],
        operations: &mut [Operation],
        pin: bool,
        single: bool,
    ) -> std::result::Result<Plan, Failure> {
        let mut working = before.to_vec();
        let mut removed = Map::new();
        let mut patched = Map::new();
        let mut after = Map::new();
        for (index, operation) in operations.iter_mut().enumerate() {
            let pos = format!("Operation {} ({})", index + 1, operation.action.name());
            if operation.action == Action::Add {
                if let Some(error) = scan::first_error(&operation.content) {
                    return Err(Failure::new(format!("{pos}: {error}")));
                }
                if !working.contains(&operation.content) {
                    working.push(operation.content.clone());
                }
                continue;
            }
            let location = if !pin {
                operation.matched_entry.as_ref().and_then(|entry| working.iter().position(|current| current == entry)).ok_or_else(|| Failure::new(format!("{pos}: Entry changed since it was staged, so this write was not applied. Recreate the change against the current entry.")))?
            } else {
                locate(&working, &operation.old_text, operation.action.name()).map_err(
                    |mut failure| {
                        if !single {
                            failure.batch(&pos, self.usage(target, before));
                        }
                        failure
                    },
                )?
            };
            let previous = working[location].clone();
            if pin {
                operation.matched_entry = Some(previous.clone());
            }
            if operation.action == Action::Remove {
                removed.insert((index + 1).to_string(), json!(previous));
                working.remove(location);
            } else {
                let count = previous
                    .char_indices()
                    .filter(|(offset, _)| previous[*offset..].starts_with(&operation.old_text))
                    .count();
                if count > 1 {
                    return Err(Failure::new(format!(
                        "'{}' appears {count} times in the matched entry, and patch edits one span. Include more surrounding text so old_text occurs once.",
                        operation.old_text
                    )));
                }
                let updated = previous.replacen(&operation.old_text, &operation.content, 1);
                if updated == previous {
                    return Err(Failure::new(format!(
                        "Patch is a no-op: '{}' -> '{}' leaves the entry unchanged.",
                        operation.old_text, operation.content
                    )));
                }
                if let Some(error) = scan::first_error(&updated) {
                    return Err(Failure::new(error));
                }
                patched.insert((index + 1).to_string(), json!(previous));
                after.insert((index + 1).to_string(), json!(updated));
                working[location] = updated;
            }
        }
        if !before.is_empty()
            && working.is_empty()
            && !(operations.len() == 1 && operations[0].action == Action::Remove)
        {
            return Err(Failure::consolidation(format!("Refusing to empty {}: this batch would remove every entry from a previously non-empty store. Keep at least one entry. To delete the final entry deliberately, remove it alone in a one-item batch.", target.filename())).with_usage(self.usage(target,before)));
        }
        let grows = !(operations.iter().all(|op| op.action == Action::Remove)
            || (operations.len() == 1 && working == before));
        if grows && char_count(&working) > self.limit(target) {
            let total = char_count(&working);
            let limit = self.limit(target);
            let mut failure = Failure::consolidation(format!("Applying {} operation(s) would put memory at {}/{limit} chars: free at least {} chars. Retry as ONE 'operations' batch that removes or shortens (patch) stale entries AND adds the new entry; the limit is checked only on the final result.", operations.len(), total, total-limit)).with_usage(self.usage(target,before));
            if single || (operations.len() == 1 && operations[0].action == Action::Add) {
                failure.value["current_entries"] = json!(before);
            } else {
                failure.value["error"] = json!(format!(
                    "{} No operations were applied (batch is all-or-nothing).",
                    failure.value["error"].as_str().unwrap_or_default()
                ));
            }
            return Err(failure);
        }
        let mut extra = Map::new();
        if single {
            if let Some(value) = removed.values().next() {
                extra.insert("removed_entry".into(), value.clone());
            }
            if let Some(value) = patched.values().next() {
                extra.insert("patched_entry_before".into(), value.clone());
            }
            if let Some(value) = after.values().next() {
                extra.insert("patched_entry_after".into(), value.clone());
            }
        } else {
            if !removed.is_empty() {
                extra.insert("removed_entries".into(), json!(removed));
            }
            if !patched.is_empty() {
                extra.insert("patched_entries".into(), json!(patched));
                extra.insert("patched_entries_after".into(), json!(after));
            }
        }
        Ok(Plan {
            entries: working,
            extra,
        })
    }

    fn enabled(&self, target: Target) -> bool {
        if target == Target::User {
            self.config.user_profile_enabled
        } else {
            self.config.memory_enabled
        }
    }
    fn limit(&self, target: Target) -> usize {
        if target == Target::User {
            self.config.user_char_limit
        } else {
            self.config.memory_char_limit
        }
    }
    fn ensure_enabled(&self, target: Target) -> Result<()> {
        if !self.enabled(target) {
            return Err(tool_error(
                Failure::new(format!(
                    "Built-in {} writes are disabled in memory config.",
                    target.filename()
                ))
                .with_field("target", json!(target.name())),
            ));
        }
        Ok(())
    }
    fn read(&self, target: Target) -> Result<String> {
        persistence::read(&self.dir, target.filename())
    }
    fn usage(&self, target: Target, entries: &[String]) -> String {
        format!(
            "{}/{}",
            commas(char_count(entries)),
            commas(self.limit(target))
        )
    }
    fn usage_pct(&self, target: Target, current: usize) -> String {
        format!(
            "{}% — {}/{} chars",
            if self.limit(target) > 0 {
                current
                    .saturating_mul(100)
                    .checked_div(self.limit(target))
                    .unwrap_or(0)
                    .min(100)
            } else {
                0
            },
            commas(current),
            commas(self.limit(target))
        )
    }
    fn render_block(&self, target: Target, entries: &[String]) -> String {
        if entries.is_empty() {
            return String::new();
        }
        let separator = "═".repeat(46);
        let title = if target == Target::User {
            "USER PROFILE (who the user is)"
        } else {
            "MEMORY (your personal notes)"
        };
        format!(
            "{separator}\n{title} [{}]\n{separator}\n{}",
            self.usage_pct(target, char_count(entries)),
            entries.join(ENTRY_DELIMITER)
        )
    }
    fn failure_result(&mut self, failure: Failure) -> Value {
        if failure.consolidation {
            self.consolidation_failures += 1;
        }
        if self.consolidation_failures > 3 {
            json!({"success":false,"done":true,"error":format!("Memory consolidation failed {} times this turn. Stop retrying memory calls — leave memory unchanged for now and continue with your reply to the user. The fact can be saved in a later turn.",self.consolidation_failures)})
        } else {
            failure.value
        }
    }
    fn count_failure(&mut self, error: anyhow::Error) -> anyhow::Error {
        if let Some(failure) = error.downcast_ref::<Failure>() {
            let result = self.failure_result(failure.clone());
            tool_error(Failure {
                value: result,
                consolidation: false,
            })
        } else {
            error
        }
    }
}

struct Plan {
    entries: Vec<String>,
    extra: Map<String, Value>,
}

#[derive(Clone, Debug)]
struct Failure {
    value: Value,
    consolidation: bool,
}
impl Failure {
    fn new(message: impl Into<String>) -> Self {
        Self {
            value: json!({"success":false,"error":message.into()}),
            consolidation: false,
        }
    }
    fn consolidation(message: impl Into<String>) -> Self {
        Self {
            consolidation: true,
            ..Self::new(message)
        }
    }
    fn with_usage(mut self, usage: String) -> Self {
        self.value["usage"] = json!(usage);
        self
    }
    fn with_field(mut self, key: &str, value: Value) -> Self {
        self.value[key] = value;
        self
    }
    fn batch(&mut self, pos: &str, usage: String) {
        self.value["error"] = json!(format!(
            "{pos}: {} No operations were applied (batch is all-or-nothing).",
            self.value["error"].as_str().unwrap_or_default()
        ));
        self.value
            .as_object_mut()
            .expect("failure object")
            .remove("current_entries");
        self.value["usage"] = json!(usage);
    }
}
impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            self.value["error"]
                .as_str()
                .unwrap_or("Memory operation failed"),
        )
    }
}
impl std::error::Error for Failure {}
fn tool_error(failure: Failure) -> anyhow::Error {
    anyhow::Error::new(failure)
}

pub fn error_result(error: &anyhow::Error) -> Value {
    error
        .downcast_ref::<Failure>()
        .map(|failure| failure.value.clone())
        .unwrap_or_else(|| json!({"success":false,"error":format!("{error:#}")}))
}

fn parse_target(value: Option<&Value>) -> std::result::Result<Target, Failure> {
    match value.and_then(Value::as_str) {
        Some("memory") => Ok(Target::Memory),
        Some("user") => Ok(Target::User),
        None => Err(Failure::new(
            "Missing required target. Retry with target='memory' or target='user' at the top level. For a batch, remove target from every operation; use separate calls for different stores. Nothing was written or staged.",
        )),
        Some(_) => Err(Failure::new(
            "Invalid memory target. Use 'memory' or 'user' at the top level.",
        )),
    }
}

fn parse_operations(
    args: &Value,
    target: Target,
) -> std::result::Result<(Vec<Operation>, bool), Failure> {
    let (raw, single) = match args.get("operations") {
        Some(Value::Array(operations)) if !operations.is_empty() => (operations.clone(), false),
        Some(Value::Null) | None | Some(Value::Array(_)) => {
            if args
                .get("action")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                return Err(Failure::new(
                    "Missing required parameter: 'operations', a list of {action: add/patch/remove, content?, old_text?} changes; a single change is a one-item list.",
                ));
            }
            (vec![args.clone()], true)
        }
        _ => {
            return Err(Failure::new(
                "operations must be a list of {action, content?, old_text?} objects.",
            ));
        }
    };
    let mut operations = Vec::with_capacity(raw.len());
    for (index, operation) in raw.iter().enumerate() {
        if !operation.is_object() {
            return Err(Failure::new(format!(
                "Operation {} must be an object.",
                index + 1
            )));
        }
        if !single
            && operation
                .get("target")
                .is_some_and(|nested| !nested.is_null() && nested != target.name())
        {
            return Err(Failure::new(format!(
                "Operation {} names a different target than the one at the top level. One call writes one store: retry with separate calls for different stores. Nothing was written or staged.",
                index + 1
            )));
        }
        let action = match operation.get("action").and_then(Value::as_str) {
            Some("add") => Action::Add,
            Some("patch") => Action::Patch,
            Some("remove") => Action::Remove,
            Some("replace") => return Err(Failure::new(REPLACE_REMOVED)),
            _ => return Err(Failure::new("Unknown action. Use: add, patch, remove")),
        };
        let value_string = |key: &str| -> std::result::Result<String, Failure> {
            match operation.get(key) {
                None | Some(Value::Null) => Ok(String::new()),
                Some(Value::String(value)) => Ok(value.clone()),
                _ => Err(Failure::new(format!("{key} must be a string."))),
            }
        };
        let mut content = value_string("content")?;
        if content.is_empty() {
            content = value_string("new_text")?;
        }
        let mut old_text = value_string("old_text")?;
        if action != Action::Patch {
            content = content.trim().into();
            old_text = old_text.trim().into();
        }
        if action != Action::Remove && content.trim().is_empty() {
            return Err(Failure::new(if action == Action::Patch {
                format!("content is required for 'patch'. {PATCH_DROP_HINT}")
            } else {
                "Content is required for 'add' action.".into()
            }));
        }
        if action != Action::Add && old_text.trim().is_empty() {
            return Err(Failure::new(format!(
                "'{}' needs old_text -- a short unique substring of the entry. Reissue with old_text set to part of the intended entry.",
                action.name()
            )));
        }
        operations.push(Operation {
            action,
            content,
            old_text,
            matched_entry: None,
        });
    }
    Ok((operations, single))
}

fn parse_entries(raw: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    raw.split(ENTRY_DELIMITER)
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .filter(|entry| seen.insert(*entry))
        .map(str::to_owned)
        .collect()
}
fn char_count(entries: &[String]) -> usize {
    entries
        .iter()
        .map(|entry| entry.chars().count())
        .sum::<usize>()
        + entries.len().saturating_sub(1) * ENTRY_DELIMITER.chars().count()
}
fn commas(value: usize) -> String {
    let s = value.to_string();
    s.chars()
        .enumerate()
        .fold(String::new(), |mut result, (index, c)| {
            if index > 0 && (s.len() - index).is_multiple_of(3) {
                result.push(',');
            }
            result.push(c);
            result
        })
}

fn normalize_match(text: &str) -> String {
    text.replace("\\n", " ")
        .chars()
        .map(|c| match c {
            '"' | '`' | '‘' | '’' | '‚' | '‛' | '“' | '”' | '„' | '‟' => '\'',
            '‐' | '‑' | '‒' | '–' | '—' | '―' | '−' => '-',
            _ => c,
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
fn locate(entries: &[String], old: &str, verb: &str) -> std::result::Result<usize, Failure> {
    if let Some(index) = entries.iter().position(|entry| entry == old) {
        return Ok(index);
    }
    let exact = entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.contains(old))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let matches = if exact.is_empty() {
        let needle = normalize_match(old);
        if !needle.chars().any(char::is_alphanumeric) {
            Vec::new()
        } else {
            entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| normalize_match(entry).contains(&needle))
                .map(|(index, _)| index)
                .collect()
        }
    } else {
        exact
    };
    if matches.len() > 1 {
        return Err(Failure::new(format!(
            "Multiple entries matched '{old}'. Be more specific."
        ))
        .with_field(
            "matches",
            json!(
                matches
                    .iter()
                    .map(|index| entries[*index].chars().take(80).collect::<String>())
                    .collect::<Vec<_>>()
            ),
        ));
    }
    if let Some(index) = matches.first() {
        return Ok(*index);
    }
    let words = |s: &str| {
        s.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|word| !word.is_empty())
            .map(str::to_lowercase)
            .collect::<HashSet<_>>()
    };
    let needle = words(old);
    let mut ranked = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| (words(entry).intersection(&needle).count(), index))
        .filter(|(score, _)| !needle.is_empty() && *score * 10 >= needle.len() * 3)
        .collect::<Vec<_>>();
    ranked.sort_by_key(|(score, index)| (std::cmp::Reverse(*score), *index));
    let mut failure=Failure::consolidation(format!("No entry matched '{old}'.{} Retry the {verb} with old_text copied verbatim from the intended entry.",if old.contains('§') {" old_text must come from ONE entry ('§' separates entries); use one op per entry."} else {""})).with_field("current_entries",json!(entries));
    if !ranked.is_empty() {
        failure.value["closest_entries"] = json!(
            ranked
                .iter()
                .take(3)
                .map(|(_, index)| &entries[*index])
                .collect::<Vec<_>>()
        );
    }
    Err(failure)
}

#[cfg(test)]
mod tests;
