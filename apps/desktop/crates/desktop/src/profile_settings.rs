//! Native settings for the current profile. Apply validates a merged settings
//! record and commits only the controls edited on this page.
use crate::appearance::text;
use super::{Hermes, Message, settings::View, ui};
use hermes_engine::settings::{PERSONALITIES,ProfileSettings};
use hermes_library::{ArchiveFilter,Library,SessionQuery,SessionSummary};
use iced::{Alignment, Element, Fill, Task, widget::{button,column,container,pick_list,row,text_input,Space}};
use std::{collections::BTreeMap,path::{Path,PathBuf},process::{Child,Command}};

#[derive(Clone,Copy,PartialEq,Eq)]
enum Kind { Text, Count, Check, Personality, Directory }
#[derive(Clone, Copy)]
struct Field { page:View, key:&'static str, label:&'static str, description:&'static str, kind:Kind }
const fn field(page:View,key:&'static str,label:&'static str,description:&'static str,kind:Kind)->Field{Field{page,key,label,description,kind}}
const FIELDS:&[Field]=&[
    field(View::Chat,"personality","Personality","Default assistant style for new sessions.",Kind::Personality),
    field(View::Chat,"timezone","Timezone","Timezone for the agent and local clock. Leave blank for the system default.",Kind::Text),
    field(View::Workspace,"default_directory","Default project folder","Working directory for new chats outside a selected project.",Kind::Directory),
    field(View::Workspace,"file_read_max_chars","File read character limit","Maximum characters returned by one file read.",Kind::Count),
    field(View::Memory,"memory_enabled","Agent memory","Let Hermes maintain persistent notes about its work.",Kind::Check),
    field(View::Memory,"memory_char_limit","Memory character limit","Maximum combined length of agent memory entries.",Kind::Count),
    field(View::Memory,"user_profile_enabled","User profile","Let Hermes keep persistent notes about your preferences.",Kind::Check),
    field(View::Memory,"user_char_limit","User profile character limit","Maximum combined length of user profile entries.",Kind::Count),
    field(View::Safety,"redact_secrets","Redact secrets","Mask detected credentials in text sent to the model.",Kind::Check),
    field(View::Safety,"approval_timeout","Approval timeout","Seconds to wait for a response before declining a tool request.",Kind::Count),
    field(View::Safety,"checkpoints_enabled","Filesystem checkpoints","Create rollback snapshots before approved native file writes.",Kind::Check),
    field(View::Safety,"checkpoint_max_snapshots","Checkpoint retention","Maximum retained file snapshots (1–100).",Kind::Count),
    field(View::Advanced,"clock_tools","System clock","Allow Hermes to request the local time tool.",Kind::Check),
    field(View::Advanced,"file_tools","File tools","Allow Hermes to request directory listing, search, read and write tools.",Kind::Check),
    field(View::Advanced,"skill_tools","Skills","Allow Hermes to discover and read workspace skills.",Kind::Check),
    field(View::Advanced,"memory_tool","Memory tool","Allow Hermes to request changes to saved memory.",Kind::Check),
    field(View::Advanced,"mcp_tools","MCP tools","Connect to the configured MCP server and offer its tools to Hermes.",Kind::Check),
    field(View::Advanced,"max_turns","Maximum iterations","Maximum tool-call rounds per agent turn.",Kind::Count),
    field(View::Advanced,"tool_output_bytes","Maximum output bytes","Size limit for each tool result.",Kind::Count),
    field(View::Advanced,"tool_output_lines","Maximum output lines","Line limit for tool output.",Kind::Count),
    field(View::Advanced,"tool_output_line_length","Maximum line length","Character limit for each line of tool output.",Kind::Count),
    #[cfg(target_os="macos")]
    field(View::Advanced,"keep_awake","Keep computer awake","Prevent idle system sleep while Hermes Desktop is open.",Kind::Check),
    field(View::Sessions,"auto_archive","Auto-archive inactive sessions","Automatically archive inactive sessions. Pinned chats stay in your session list.",Kind::Check),
    field(View::Sessions,"auto_archive_days","Days of inactivity","How long to wait before archiving an inactive chat.",Kind::Count),
];
#[derive(Clone,Debug)]
pub(super) enum Action {
    Set(&'static str,String,View), Edit(&'static str,String),Apply(View),Browse,DirectoryChosen(Result<Option<PathBuf>,String>),
    Restore(String),Delete(String),CancelDelete,ConfirmDelete,Maintain,
}
#[derive(Default)]
pub(super) struct State {
    pub saved:ProfileSettings,
    edits:BTreeMap<&'static str,String>,
    pub archive:Vec<SessionSummary>,
    pub feedback:Option<Result<String,String>>,
    delete:Option<String>,
    awake:Option<Child>,
}
impl Drop for State{fn drop(&mut self){self.release_awake();}}
impl State {
    fn release_awake(&mut self){if let Some(mut child)=self.awake.take(){let _=child.kill();let _=child.wait();}}
    fn sync_awake(&mut self)->Result<(),String>{
        #[cfg(target_os="macos")]
        if self.saved.keep_awake && self.awake.is_none(){
            self.awake=Some(Command::new("/usr/bin/caffeinate").args(["-i","-w",&std::process::id().to_string()]).spawn().map_err(|e|e.to_string())?);return Ok(());
        }
        if !self.saved.keep_awake {self.release_awake();}
        Ok(())
    }
    pub fn load(root:&Path)->Self{
        let mut state=Self::default();
        match ProfileSettings::load(root) {
            Ok(saved)=>state.saved=saved, Err(e)=>state.feedback=Some(Err(e.to_string())),
        }
        if let Err(e)=state.sync_awake(){state.feedback=Some(Err(e));}
        state
    }
    pub fn archived(&mut self,lib:&Library){
        match lib.list_sessions(&SessionQuery{archive:ArchiveFilter::Archived,..Default::default()}){
            Ok(archive)=>self.archive=archive,Err(e)=>self.feedback=Some(Err(e.to_string())),
        }
    }
    fn value(&self,f:&Field)->String{
        if let Some(v)=self.edits.get(f.key){return v.clone();}
        let v=serde_json::to_value(&self.saved).unwrap_or_default();
        match &v[f.key] {serde_json::Value::Null=>String::new(),serde_json::Value::String(s)=>s.clone(),other=>other.to_string()}
    }
    fn apply(&mut self,page:View,root:&Path)->Result<(),String>{
        let mut changes=serde_json::Map::new();
        for f in FIELDS.iter().filter(|f|f.page==page){
            let Some(entry)=self.edits.get(f.key) else{continue;};
            let entry=entry.trim();
            changes.insert(f.key.to_string(),match f.kind {
                Kind::Count=>serde_json::json!(entry.parse::<u64>().map_err(|_|format!("Enter a whole number for {}.",f.label))?),
                Kind::Check=>serde_json::json!(entry.parse::<bool>().map_err(|_|"Invalid setting")?),
                Kind::Directory if entry.is_empty()=>serde_json::Value::Null,
                Kind::Directory=>{let path=super::expand_path(entry).canonicalize().map_err(|_|"Choose an existing project folder")?;
                    if !path.is_dir(){return Err("Choose a project folder".into())}serde_json::json!(path)}
                _=>serde_json::json!(entry),
            });
        }
        let next=ProfileSettings::update(root,&changes).map_err(|e|format!("{e:#}"))?;
        self.saved=next;self.edits.retain(|k,_|!FIELDS.iter().any(|f|f.key==*k && f.page==page));
        self.sync_awake()
    }
}
pub(super) fn update(app:&mut Hermes,action:Action)->Task<Message>{
    let Some(lib)=app.library.clone() else{return Task::none()};
    let state=&mut app.profile_settings;
    match action {
        // Switches and pickers save on change, as the original's autosave does.
        Action::Set(key,value,page)=>{state.edits.insert(key,value);return update(app,Action::Apply(page));}
        Action::Edit(key,value)=>{state.edits.insert(key,value);state.feedback=None;}
        Action::Apply(page)=>{
            let result=state.apply(page,lib.root());let applied=result.is_ok();state.feedback=Some(result.map(|_|"Settings saved".into()));
            if applied && page==View::Sessions{return update(app,Action::Maintain)}
        }
        Action::Browse=>return Task::perform(async {tokio::task::spawn_blocking(super::choose_folder).await.map_err(|e|e.to_string()).and_then(|r|r)},|r|Message::ProfileSetting(Action::DirectoryChosen(r))),
        Action::DirectoryChosen(Ok(Some(folder)))=>return update(app,Action::Edit("default_directory",folder.display().to_string())),
        Action::DirectoryChosen(Err(e))=>state.feedback=Some(Err(e)),
        Action::DirectoryChosen(Ok(None))=>{}
        Action::Maintain=>{
            if state.saved.auto_archive {
                if let Err(e)=lib.archive_inactive(state.saved.auto_archive_days,app.current.as_ref().map(|s|s.id.as_str())){state.feedback=Some(Err(e.to_string()));}
            }
            if app.settings_view==View::Sessions{state.archived(&lib);}
            app.refresh_sessions();
        }
        Action::Restore(id)=>{
            state.feedback=Some(lib.set_archived(&id,false).map(|_|"Session restored".into()).map_err(|e|e.to_string()));
            state.archived(&lib);app.refresh_sessions();
        }
        Action::Delete(id)=>state.delete=Some(id),
        Action::CancelDelete=>state.delete=None,
        Action::ConfirmDelete=>if let Some(id)=state.delete.take(){
            state.feedback=Some(lib.delete_archived(&id).map(|_|"Session deleted".into()).map_err(|e|e.to_string()));
            state.archived(&lib);app.refresh_sessions();
        }
    }
    Task::none()
}
pub(super) fn view(app:&Hermes,page:View)->Element<'_,Message>{
    let p=ui::palette(app.dark);let state=&app.profile_settings;
    let title=match page {View::Chat=>"Behavior",View::Workspace=>"Projects & files",View::Memory=>"Persistent memory",View::Safety=>"Approvals",View::Advanced=>"Runtime & tool output",View::Sessions=>"Sessions",_=>""};
    let dark=app.dark;
    let mut body=column![super::settings::section_title(title,dark)].spacing(8);
    let mut rows=Vec::new();
    for f in FIELDS.iter().filter(|f|f.page==page){
        let key=f.key;let value=state.value(f);
        let control:Element<'_,Message>=match f.kind {
            Kind::Check=>super::settings::switch(value=="true",move|b|Message::ProfileSetting(Action::Set(key,b.to_string(),page)),dark),
            Kind::Personality=>{
                let options=std::iter::once("default".to_owned()).chain(PERSONALITIES.keys().cloned()).collect::<Vec<_>>();
                pick_list(options,Some(value),move|s|Message::ProfileSetting(Action::Set(key,s,page))).width(180).text_size(13).padding([7,10]).into()
            }
            _=>{
                let input=text_input(if key=="timezone"{"System default"}else{""},&value).size(13).padding([7,10]).width(if f.kind==Kind::Count{110}else{220})
                .style(super::settings::field_style(dark))
                .on_input(move|s|Message::ProfileSetting(Action::Edit(key,s))).on_submit(Message::ProfileSetting(Action::Apply(page)));
                if f.kind==Kind::Directory {row![input,button(text("Browse…").size(13)).padding([7,10]).on_press(Message::ProfileSetting(Action::Browse))].spacing(6).into()}else{input.into()}
            }
        };
        rows.push(super::settings::card_row(f.label,f.description,control,dark));
    }
    body=body.push(super::settings::card(rows,dark));
    // Text fields save on Enter; Apply is offered only while one has unsaved edits.
    let dirty=FIELDS.iter().any(|f|f.page==page && matches!(f.kind,Kind::Count|Kind::Text|Kind::Directory) && state.edits.contains_key(f.key));
    if dirty {body=body.push(row![text("Unsaved changes").size(12).color(p.muted),Space::new().width(Fill),button(text("Save").size(13)).padding([6,14]).on_press(Message::ProfileSetting(Action::Apply(page)))].align_y(Alignment::Center));}
    if let Some(result)=&state.feedback{body=body.push(match result {Ok(s)=>text(s).color(p.muted),Err(s)=>text(s).color(iced::Color::from_rgb8(194,61,64))}.size(12));}
    if page==View::Workspace {body=body.push(Space::new().height(12)).push(super::project_discovery::view(app));}
    if page==View::Safety {body=body.push(super::checkpoint_ui::view(app));}
    if page==View::Sessions{
        body=body.push(text("Archived sessions").size(16));
        if state.archive.is_empty(){body=body.push(text("No archived sessions").size(13).color(p.muted));}
        for chat in &state.archive{
            let (restore,remove)=(chat.id.clone(),chat.id.clone());
            body=body.push(row![text(&chat.title).size(13).width(Fill),
                button(text("Restore").size(12)).on_press(Message::ProfileSetting(Action::Restore(restore))),
                button(text("Delete…").size(12)).on_press(Message::ProfileSetting(Action::Delete(remove)))].spacing(6));
            if state.delete.as_ref()==Some(&chat.id) {
                body=body.push(row![text("Permanently delete this archived conversation?").size(12),
                button(text("Cancel").size(12)).on_press(Message::ProfileSetting(Action::CancelDelete)),
                button(text("Delete permanently").size(12)).on_press(Message::ProfileSetting(Action::ConfirmDelete))].spacing(6));
            }
        }
        body=body.push(button(text("Import sessions…").size(13)).on_press(Message::Navigate(super::Screen::Import)));
    }
    super::settings::settings_content(container(body).width(Fill).into())
}
