//! First-party text skills and the user's explicitly installed documents.
//! Catalog IDs are opaque map keys, never filesystem paths or shell commands.
use anyhow::{Context,Result,ensure};
use flate2::read::GzDecoder;
use serde::{Deserialize,Serialize};
use serde_json::{Value,json};
use std::{collections::{BTreeMap,BTreeSet},fs,io::Write,path::Path,sync::LazyLock};

static DOCUMENTS:LazyLock<BTreeMap<String,String>>=LazyLock::new(||serde_json::from_reader(GzDecoder::new(include_bytes!("../assets/catalog/documents.json.gz").as_slice())).expect("bundled Hermes skills"));
#[derive(Default,Deserialize,Serialize)]
#[serde(default)]
struct Store{disabled:BTreeSet<String>,installed:BTreeMap<String,String>}
#[derive(Clone,Default)]
pub struct Status{pub installed:BTreeSet<String>,pub disabled:BTreeSet<String>}
fn read(root:&Path)->Result<Store>{
    let path=root.join("desktop-skills.json");
    match fs::symlink_metadata(&path){
        Ok(meta)=>{ensure!(meta.is_file()&&meta.len()<32_000_000,"Invalid skill library");Ok(serde_json::from_reader(fs::File::open(path)?)?)}
        Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(Store::default()),Err(e)=>Err(e.into())
    }
}
fn is_builtin(id:&str)->bool{id.starts_with("NousResearch/hermes-agent/skills/") && DOCUMENTS.contains_key(id)}
pub fn can_install(id:&str)->bool{DOCUMENTS.contains_key(id)||raw_url(id).is_some()}

/// One row of the published skills snapshot (`docs/api/skills.json`).
#[derive(Clone,Debug)]
pub struct Entry{pub id:String,pub name:String,pub description:String,pub category:String,pub source:String,pub author:String,pub version:String,pub docs:String,pub platforms:Vec<String>,pub featured:usize}
static ENTRIES:LazyLock<Vec<Entry>>=LazyLock::new(||{
    let rows:Vec<Vec<Value>>=serde_json::from_reader(GzDecoder::new(include_bytes!("../assets/catalog/skills.json.gz").as_slice())).unwrap_or_default();
    let text=|v:Option<&Value>|v.and_then(Value::as_str).unwrap_or_default().to_owned();
    rows.into_iter().filter(|r|r.len()>=5).map(|r|{
        let id=text(r.first());
        let featured=FEATURED.iter().position(|key|id==format!("NousResearch/hermes-agent/skills/{key}")).map_or(0,|i|i+1);
        Entry{name:text(r.get(1)),description:text(r.get(2)),category:text(r.get(3)),source:text(r.get(4)),author:text(r.get(5)),version:text(r.get(6)),docs:text(r.get(7)),
            platforms:r.get(8).and_then(Value::as_array).map(|a|a.iter().filter_map(|v|v.as_str().map(str::to_owned)).collect()).unwrap_or_default(),featured,id}
    }).collect()
});
/// `shared/src/catalog-curation.json`: the editorially featured skills, in hero order.
pub const FEATURED:[&str;4]=["autonomous-ai-agents/hermes-agent","autonomous-ai-agents/computer-use","autonomous-ai-agents/codex","autonomous-ai-agents/claude-code"];
/// The whole published catalog, decoded once on first use (call off the UI thread).
pub fn entries()->&'static [Entry]{&ENTRIES}
/// The published plugin snapshot (`docs/api/plugins.json`): `[name, description, category, url]` rows.
pub fn plugin_rows()->Vec<Vec<Value>>{
    serde_json::from_reader(GzDecoder::new(include_bytes!("../assets/catalog/plugins.json.gz").as_slice())).unwrap_or_default()
}
pub fn is_official(source:&str)->bool{matches!(source,"built-in"|"optional")}
/// Where a skill's SKILL.md is published, for sources that are plain GitHub repositories.
/// Hub-only sources (ClawHub, LobeHub, skills.sh, browse.sh) return `None`.
fn raw_url(id:&str)->Option<String>{
    let safe=|part:&str|!part.is_empty()&&part!=".."&&part.chars().all(|c|c.is_ascii_alphanumeric()||"-_.".contains(c));
    let parts:Vec<&str>=id.split('/').collect();
    if !parts.iter().all(|p|safe(p)){return None}
    if let ["official",category,name]=parts.as_slice(){
        return Some(format!("https://raw.githubusercontent.com/NousResearch/hermes-agent/HEAD/optional-skills/{category}/{name}/SKILL.md"));
    }
    let hub=["clawhub","lobehub","skills-sh","browse-sh"];
    if parts.len()<3||hub.contains(&parts[0]){return None}
    Some(format!("https://raw.githubusercontent.com/{}/{}/HEAD/{}/SKILL.md",parts[0],parts[1],parts[2..].join("/")))
}
fn state(store:&Store)->Status{
    Status{disabled:store.disabled.clone(),installed:DOCUMENTS.keys().filter(|id|is_builtin(id)).cloned().chain(store.installed.keys().cloned()).collect()}
}
pub fn status(root:&Path)->Result<Status>{Ok(state(&read(root)?))}
fn modify(root:&Path,action:impl FnOnce(&mut Store)->Result<()>)->Result<Status>{
    let guard=fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("desktop-skills.lock"))?;
    fs2::FileExt::lock_exclusive(&guard)?;
    let mut current=read(root)?;action(&mut current)?;
    let mut file=tempfile::NamedTempFile::new_in(root)?;serde_json::to_writer(&mut file,&current)?;file.flush()?;file.as_file().sync_all()?;file.persist(root.join("desktop-skills.json"))?;
    Ok(state(&current))
}
pub fn enable(root:&Path,id:&str,on:bool)->Result<Status>{modify(root,|s|{ensure!(is_builtin(id)||s.installed.contains_key(id),"Install this skill first");if on{s.disabled.remove(id);}else{s.disabled.insert(id.to_owned());}Ok(())})}
pub fn remove(root:&Path,id:&str)->Result<Status>{modify(root,|s|{ensure!(!is_builtin(id),"Built-in skills can be switched off");s.installed.remove(id);s.disabled.remove(id);Ok(())})}
pub async fn install(root:&Path,id:&str)->Result<Status>{
    let doc=if let Some(doc)=DOCUMENTS.get(id){doc.clone()}else{
        let url=raw_url(id).context("This skill installs from its publisher's hub; open its page to add it")?;
        let mut resp=reqwest::Client::builder().redirect(reqwest::redirect::Policy::none()).timeout(std::time::Duration::from_secs(25)).build()?
            .get(url).send().await?.error_for_status().context("The publisher did not return this skill")?;
        let mut bytes=vec![];while let Some(chunk)=resp.chunk().await?{ensure!(bytes.len()+chunk.len()<=512*1024,"Skill document too large");bytes.extend_from_slice(&chunk);}
        let doc=String::from_utf8(bytes)?;ensure!(doc.starts_with("---"),"Publisher returned an invalid skill document");doc
    };
    modify(root,|s|{if !is_builtin(id){s.installed.insert(id.into(),doc);}s.disabled.remove(id);Ok(())})
}
/// Replace every installed document under `prefix` with `docs` (plugin skills).
pub fn put_documents(root:&Path,prefix:&str,docs:Vec<(String,String)>)->Result<Status>{modify(root,|s|{s.installed.retain(|id,_|!id.starts_with(prefix));s.disabled.retain(|id|!id.starts_with(prefix));for (id,doc) in docs{s.installed.insert(id,doc);}Ok(())})}
pub fn document(root:&Path,id:&str)->Result<Option<String>>{
    let s=read(root)?;Ok(s.installed.get(id).or_else(||DOCUMENTS.get(id)).cloned())
}
pub(crate) fn merge(root:&Path,name:&str,args:&Value,local:Result<Value>)->Result<Value>{
    if name=="skills_list"{
        let mut result=local?;let rows=result["skills"].as_array_mut().context("Invalid workspace skills list")?;
        let s=read(root)?;let st=state(&s);
        for id in st.installed.difference(&st.disabled){
            let parts:Vec<_>=id.split('/').collect();let slug=*parts.last().unwrap_or(&"");
            let category=parts.iter().rev().nth(1).unwrap_or(&"general");
            if args["category"].as_str().is_some_and(|c|!c.eq_ignore_ascii_case(category)){continue;}
            let doc=s.installed.get(id).or_else(||DOCUMENTS.get(id));let description=doc.and_then(|d|d.lines().take(18).find_map(|l|l.strip_prefix("description:"))).map(|v|v.trim().trim_matches('"')).unwrap_or("");
            rows.push(json!({"id":format!("catalog:{id}"),"name":slug,"category":category,"description":description}));
        }
        return Ok(result)
    }
    if name=="skill_view"{
        let requested=args["name"].as_str().unwrap_or_default();
        let s=read(root)?;let st=state(&s);
        let id=if let Some(id)=requested.strip_prefix("catalog:"){Some(id)}else if local.is_err(){
            let mut matches=st.installed.iter().filter(|id|id.rsplit('/').next()==Some(requested));
            let first=matches.next().map(String::as_str);if matches.next().is_some(){None}else{first}
        }else{None};
        if let Some(id)=id{
            ensure!(st.installed.contains(id)&&!st.disabled.contains(id),"This skill is disabled or missing");
            ensure!(args["file_path"].is_null(),"A supporting file for this catalog document is unavailable");
            let doc=s.installed.get(id).or_else(||DOCUMENTS.get(id)).context("Missing installed document")?;
            let offset=args["offset"].as_u64().unwrap_or(1).max(1) as usize;let limit=(args["limit"].as_u64().unwrap_or(160) as usize).clamp(1,1000);
            let lines=doc.lines().collect::<Vec<_>>();let start=offset.saturating_sub(1).min(lines.len());let end=start.saturating_add(limit).min(lines.len());
            return Ok(json!({"id":id,"content":lines[start..end].join("\n"),"next_offset":(end<lines.len()).then_some(end+1)}))
        }
    }
    local
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn catalog_decodes_and_maps_install_sources(){
        let all=entries();
        assert!(all.len()>100_000);
        assert_eq!(all.iter().filter(|e|e.featured>0).count(),FEATURED.len());
        assert!(all.iter().any(|e|e.source=="built-in"&&e.name=="apple-notes"));
        assert_eq!(raw_url("official/autonomous-ai-agents/blackbox").as_deref(),Some("https://raw.githubusercontent.com/NousResearch/hermes-agent/HEAD/optional-skills/autonomous-ai-agents/blackbox/SKILL.md"));
        assert_eq!(raw_url("anthropics/skills/skills/algorithmic-art").as_deref(),Some("https://raw.githubusercontent.com/anthropics/skills/HEAD/skills/algorithmic-art/SKILL.md"));
        assert!(raw_url("clawhub/apple-docs").is_none());
        assert!(raw_url("a/../b/c").is_none());
        assert!(can_install("NousResearch/hermes-agent/skills/apple/apple-notes"));
    }
    #[test]
    fn disabled_skill_is_hidden_from_the_agent(){
        let dir=tempfile::tempdir().unwrap();
        let id="NousResearch/hermes-agent/skills/apple/apple-notes";
        let listed=|dir:&Path|merge(dir,"skills_list",&json!({}),Ok(json!({"skills":[]}))).unwrap()["skills"].as_array().unwrap().iter().any(|s|s["id"]==format!("catalog:{id}"));
        assert!(listed(dir.path()));
        enable(dir.path(),id,false).unwrap();
        assert!(!listed(dir.path()));
        assert!(merge(dir.path(),"skill_view",&json!({"name":format!("catalog:{id}")}),Ok(json!({}))).is_err());
        enable(dir.path(),id,true).unwrap();
        assert!(listed(dir.path()));
    }
}
