//! Profile-scoped controls used by native sessions, including sessions opened
//! through the gateway. The field names correspond to the desktop settings.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::{Path,PathBuf}, sync::LazyLock, collections::BTreeMap};

pub static PERSONALITIES: LazyLock<BTreeMap<String,String>> = LazyLock::new(||
    serde_json::from_str(include_str!("personalities.json")).expect("built-in personality resources"));
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ProfileSettings {
    pub checkpoints_enabled: bool,
    pub checkpoint_max_snapshots: usize,
    pub redact_secrets: bool,
    pub clock_tools: bool,
    pub file_tools: bool,
    pub skill_tools: bool,
    pub memory_tool: bool,
    pub mcp_tools: bool,
    pub personality: String,
    pub timezone: String,
    pub default_directory: Option<PathBuf>,
    pub memory_enabled: bool,
    pub user_profile_enabled: bool,
    pub memory_char_limit: usize,
    pub user_char_limit: usize,
    pub approval_timeout: u64,
    /// `manual` asks before every tool; `smart` runs read-only tools and asks before
    /// changes; `off` never asks (approval-mode-menu.tsx).
    pub approval_mode: String,
    pub max_turns: usize,
    pub file_read_max_chars:usize,
    pub tool_output_bytes: usize,
    pub tool_output_lines: usize,
    pub tool_output_line_length: usize,
    pub auto_archive: bool,
    pub auto_archive_days: u32,
    pub keep_awake: bool,
}
impl Default for ProfileSettings {
    fn default()->Self {Self {checkpoints_enabled:false,checkpoint_max_snapshots:20,redact_secrets:true,clock_tools:true,file_tools:true,skill_tools:true,memory_tool:true,mcp_tools:true, personality:"default".into(), timezone:String::new(), default_directory:None,
        memory_enabled:true,user_profile_enabled:true,memory_char_limit:crate::memory::DEFAULT_MEMORY_CHAR_LIMIT,
        user_char_limit:crate::memory::DEFAULT_USER_CHAR_LIMIT, approval_timeout:300, approval_mode:"smart".into(), max_turns:8,
        file_read_max_chars:100_000,tool_output_bytes:50000,tool_output_lines:2000,tool_output_line_length:2000,
        auto_archive:false,auto_archive_days:3,keep_awake:false }}
}
impl ProfileSettings {
    pub fn load(root:&Path)->Result<Self>{
        match fs::File::open(root.join("agent-settings.json")) {
            Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(Self::default()),
            file=>{let value:Self=serde_json::from_reader(file?)?;value.validate()?;Ok(value)}
        }
    }
    pub fn validate(&self)->Result<()>{
        ensure!((1..=100).contains(&self.checkpoint_max_snapshots),"Checkpoint retention must be between 1 and 100 snapshots");
        ensure!(self.personality=="default" || PERSONALITIES.contains_key(&self.personality),"Unknown assistant personality");
        ensure!(self.timezone.is_empty() || self.timezone.parse::<chrono_tz::Tz>().is_ok(),"Enter an IANA timezone, for example Australia/Sydney");
        if let Some(dir)=&self.default_directory {ensure!(dir.is_absolute(),"The default project folder must be an absolute path");}
        ensure!((1..=1_000_000).contains(&self.memory_char_limit) && (1..=1_000_000).contains(&self.user_char_limit),"Memory limits must be between 1 and 1,000,000 characters");
        ensure!(["manual","smart","off"].contains(&self.approval_mode.as_str()),"Approval mode must be manual, smart or off");
        ensure!((1..=86400).contains(&self.approval_timeout),"Approval timeout must be between 1 and 86,400 seconds");
        ensure!((1..=1000).contains(&self.max_turns),"Maximum iterations must be between 1 and 1,000");
        ensure!((1..=1_000_000).contains(&self.file_read_max_chars),"File read limit must be between 1 and 1,000,000 characters");
        ensure!((512..=4_194_304).contains(&self.tool_output_bytes),"Tool output size must be between 512 bytes and 4 MiB");
        ensure!((1..=100_000).contains(&self.tool_output_lines) && (32..=1_000_000).contains(&self.tool_output_line_length),"Enter 1–100,000 output lines and 32–1,000,000 characters per line");
        ensure!((1..=36500).contains(&self.auto_archive_days),"Auto-archive requires a positive number of days (up to 36,500)");
        Ok(())
    }
    pub fn update(root:&Path, changes:&serde_json::Map<String, serde_json::Value>)->Result<Self>{
        let lock=fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("agent-settings.lock"))?;
        fs2::FileExt::lock_exclusive(&lock)?;
        let mut current=serde_json::to_value(Self::load(root)?)?;
        let fields=current.as_object_mut().context("Invalid profile settings")?;
        for (key,value) in changes {
            ensure!(fields.contains_key(key),"Unknown profile setting {key}");
            fields.insert(key.clone(),value.clone());
        }
        let next:Self=serde_json::from_value(current)?;
        next.save(root)?;
        Ok(next)
    }
    fn save(&self,root:&Path)->Result<()> {
        self.validate()?;
        let mut file=tempfile::NamedTempFile::new_in(root)?;
        serde_json::to_writer_pretty(&mut file,self)?; file.write_all(b"\n")?;file.as_file().sync_all()?;
        file.persist(root.join("agent-settings.json")).context("Could not save profile settings")?;
        Ok(())
    }
    /// Whether a tool call must wait for the user's answer under the approval mode.
    pub fn needs_approval(&self,name:&str,mutates:bool,external:bool)->bool {
        match self.approval_mode.as_str() {
            "off"=>false,
            "smart"=>mutates||external||name=="memory",
            _=>true,
        }
    }
    pub fn tool_enabled(&self,name:&str)->bool {
        match name {
            "get_current_time"=>self.clock_tools,
            "list_directory"|"read_file"|"write_file"|"search_files"=>self.file_tools,
            "skills_list"|"skill_view"=>self.skill_tools,
            "memory"=>self.memory_tool,
            "delegate_task"=>true,
            _=>self.mcp_tools,
        }
    }
    pub(crate) fn memory_config(&self,scheduled:bool)->crate::memory::MemoryConfig{
        crate::memory::MemoryConfig{memory_enabled:self.memory_enabled&&!scheduled,user_profile_enabled:self.user_profile_enabled&&!scheduled,memory_char_limit:self.memory_char_limit,user_char_limit:self.user_char_limit}
    }
    pub(crate) fn bound_tool_result(&self,text:String)->String {
        let Ok(mut data)=serde_json::from_str::<serde_json::Value>(&text) else {return text;};
        fn clip(value:&mut serde_json::Value, settings:&ProfileSettings, clipped:&mut bool){
            match value {
                serde_json::Value::String(content) => {
                    if content.lines().count()<=settings.tool_output_lines && content.lines().all(|line|line.chars().count()<=settings.tool_output_line_length){return}
                    *content=content.lines().take(settings.tool_output_lines).map(|line|line.chars().take(settings.tool_output_line_length).collect::<String>()).collect::<Vec<_>>().join("\n");
                    *clipped=true;
                }
                serde_json::Value::Array(values)=>for value in values{clip(value,settings,clipped)},
                serde_json::Value::Object(values)=>for value in values.values_mut(){clip(value,settings,clipped)},
                _=>{}
            }
        }
        let mut clipped=false;clip(&mut data,self,&mut clipped);
        let encoded=if clipped {serde_json::json!({"truncated":true,"result":data}).to_string()}else{text};
        if encoded.len()<=self.tool_output_bytes { return encoded; }
        // Measure the escaped prefix against the serialized envelope. A quote,
        // backslash or control character can expand when embedded in JSON.
        let mut result=serde_json::json!({"truncated":true,"output":"","original_bytes":encoded.len()});
        let available=self.tool_output_bytes.saturating_sub(result.to_string().len());
        let mut used=0;
        let mut prefix=String::new();
        for ch in encoded.chars() {
            let bytes=match ch { '"' | '\\' | '\n' | '\r' | '\t' | '\u{0008}' | '\u{000c}'=>2, c if c<' '=>6, c=>c.len_utf8() };
            if used+bytes>available {break;}
            prefix.push(ch);used+=bytes;
        }
        result["output"]=prefix.into();
        result.to_string()
    }
}

#[cfg(test)]
mod approval_tests {
    use super::*;
    #[test]
    fn approval_modes() {
        let mut s=ProfileSettings::default();
        assert_eq!(s.approval_mode,"smart");
        assert!(!s.needs_approval("read_file",false,false));
        assert!(s.needs_approval("write_file",true,false));
        assert!(s.needs_approval("remote_tool",false,true));
        assert!(s.needs_approval("memory",false,false));
        s.approval_mode="manual".into(); assert!(s.needs_approval("get_current_time",false,false));
        s.approval_mode="off".into(); assert!(!s.needs_approval("write_file",true,true));
        s.approval_mode="sometimes".into(); assert!(s.validate().is_err());
    }
}
