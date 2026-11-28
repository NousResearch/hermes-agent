//! Local session tabs. Geometry and verbs follow stock PaneTab/PaneTabStrip
//! and tree/renderer/tree-group.tsx at 0284ce17. Closing a tab does not delete
//! its session, draft, queued messages, or attachments.
use hermes_engine::History;
use hermes_library::{ArchiveFilter, Library, SessionQuery, SessionSource, SessionSummary};
use serde::{Deserialize, Serialize};
use std::{collections::{HashMap, HashSet}, fs, io::Write, path::Path};
use tempfile::NamedTempFile;

#[derive(Clone)]
pub(super) struct Tab { pub id: String, pub title: String, pub draft: bool }
#[derive(Clone, Default)]
pub(super) struct State {
    pub entries: Vec<Tab>, pub active: Option<String>, pub hover: Option<String>,
    pub menu: Option<String>, pub dragging: Option<String>, closed: Vec<String>,
}
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct Saved { order: Vec<String>, active: Option<String>, closed: Vec<String> }

#[derive(Clone, Debug)]
pub(super) enum Action {
    Press(String), Select(String), Close(String), CloseOthers(String), CloseToRight(String), CloseAll,
    Reopen, Next(bool), Menu(String), Hover(Option<String>), Release, Dismiss,
}

impl State {
    pub fn load(library:&Library)->Self {
        let saved:Saved=fs::read(library.root().join("desktop-tabs.json")).ok().and_then(|bytes|serde_json::from_slice(&bytes).ok()).unwrap_or_default();
        let summaries=library.list_sessions(&SessionQuery{archive:ArchiveFilter::All,..Default::default()}).unwrap_or_default();
        let summaries:HashMap<_,_>=summaries.into_iter().map(|summary|(summary.id.clone(),summary)).collect();
        let mut state=Self::default();let mut seen=HashSet::new();
        for id in saved.order.into_iter().take(128) {
            if !seen.insert(id.clone()){continue}
            if let Some(summary)=summaries.get(&id) && !summary.archived {
                state.entries.push(Tab {id,title:summary.title.clone(),draft:empty_native(summary)});
            }
        }
        state.active=saved.active.filter(|id|state.entries.iter().any(|tab|&tab.id==id));
        state.closed=saved.closed.into_iter().filter(|id|summaries.contains_key(id)&&!state.entries.iter().any(|tab|&tab.id==id)).take(30).collect();
        state
    }
    pub fn save(&self,root:&Path)->Result<(),String> {
        let saved=Saved{order:self.entries.iter().map(|tab|tab.id.clone()).collect(),active:self.active.clone(),closed:self.closed.clone()};
        let mut file=NamedTempFile::new_in(root).map_err(|e|e.to_string())?;
        serde_json::to_writer(&mut file,&saved).map_err(|e|e.to_string())?;
        file.flush().and_then(|_|file.as_file().sync_all()).map_err(|e|e.to_string())?;
        file.persist(root.join("desktop-tabs.json")).map_err(|e|e.to_string())?;Ok(())
    }
    pub fn activate(&mut self,summary:&SessionSummary,draft:bool) {
        if let Some(tab)=self.entries.iter_mut().find(|tab|tab.id==summary.id){tab.title=summary.title.clone();tab.draft=draft;}
        else{self.entries.push(Tab{id:summary.id.clone(),title:summary.title.clone(),draft});}
        self.active=Some(summary.id.clone());self.menu=None;
        self.closed.retain(|id|id!=&summary.id);
    }
    pub fn summary(&mut self,summary:&SessionSummary) {
        if let Some(tab)=self.entries.iter_mut().find(|tab|tab.id==summary.id){tab.title=summary.title.clone();if summary.title!="New chat"||!matches!(summary.source,SessionSource::Native|SessionSource::LegacyRust){tab.draft=false;}}
    }
    pub fn committed(&mut self,id:&str) {if let Some(tab)=self.entries.iter_mut().find(|tab|tab.id==id){tab.draft=false;}}
    pub fn close(&mut self,id:&str)->Option<String> {
        let index=self.entries.iter().position(|tab|tab.id==id)?;
        let tab=self.entries.remove(index);self.closed.retain(|closed|closed!=&tab.id);self.closed.push(tab.id);if self.closed.len()>30{self.closed.remove(0);}
        self.menu=None;self.dragging=None;
        if self.active.as_deref()==Some(id){self.active=self.entries.get(index.saturating_sub(1)).or_else(||self.entries.first()).map(|tab|tab.id.clone());}
        self.active.clone()
    }
    pub fn next(&self,forward:bool)->Option<String> {
        if self.entries.is_empty(){return None}
        let index=self.active.as_ref().and_then(|id|self.entries.iter().position(|tab|&tab.id==id)).unwrap_or(0);
        let next=if forward{(index+1)%self.entries.len()}else{(index+self.entries.len()-1)%self.entries.len()};Some(self.entries[next].id.clone())
    }
    pub fn reopen(&mut self)->Option<String> {while let Some(id)=self.closed.pop(){if !self.entries.iter().any(|tab|tab.id==id){return Some(id)}}None}
    pub fn move_to(&mut self,id:&str,target:&str)->bool {
        let Some(from)=self.entries.iter().position(|tab|tab.id==id)else{return false};let Some(to)=self.entries.iter().position(|tab|tab.id==target)else{return false};if from==to{return false}
        let tab=self.entries.remove(from);self.entries.insert(to.min(self.entries.len()),tab);true
    }
    pub fn reusable(&self,library:&Library,project:Option<&str>)->Option<SessionSummary> {
        self.entries.iter().filter(|tab|tab.draft).find_map(|tab|library.session(&tab.id).ok().filter(|summary|summary.project_id.as_deref()==project&&pristine(summary)))
    }
}

pub(super) fn empty_native(summary:&SessionSummary)->bool {
    if summary.title!="New chat"||!matches!(summary.source,SessionSource::Native|SessionSource::LegacyRust){return false}
    fs::read(summary.directory.join("conversation.json")).ok().and_then(|bytes|serde_json::from_slice::<History>(&bytes).ok()).is_some_and(|history|history.turns.is_empty())
}
pub(super) fn pristine(summary:&SessionSummary)->bool {
    empty_native(summary)
        && fs::read_to_string(summary.directory.join("draft.txt")).map(|draft|draft.trim().is_empty()).unwrap_or(true)
        && fs::read(summary.directory.join("draft-attachments.json")).ok().and_then(|bytes|serde_json::from_slice::<Vec<serde_json::Value>>(&bytes).ok()).is_none_or(|files|files.is_empty())
        && !summary.directory.join("queued-prompts.json").exists()
}

/// Uppercase 11px labels run ~7.6px a character, plus the status dot, close button and padding.
pub(super) fn tab_width(tab:&Tab)->f32 {
    let chars = if tab.draft { "NEW SESSION".len() } else { tab.title.chars().count().min(19) };
    (chars as f32 * 7.6 + 64.).clamp(112., 210.)
}
