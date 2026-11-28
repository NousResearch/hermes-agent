//! Profile override of the existing HTTP MCP transport, scoped to the exact
//! configured URL. Environment-based setups keep working without migration.
use crate::mcp::Mcp;
use anyhow::{Context,Result,ensure};
use reqwest::Url;
use serde::{Deserialize,Serialize};
use std::{fs, io::Write, path::Path};

#[derive(Default,Deserialize,Serialize)]
struct Endpoint {url:String,token:Option<String>}
#[derive(Clone,Default)]
pub struct Summary{pub url:String,pub has_token:bool,pub environment:bool}
fn read(root:&Path)->Result<Option<Endpoint>>{
    let path=root.join("mcp-endpoint.json");
    match fs::symlink_metadata(&path){
        Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Ok(None),
        Err(e)=>Err(e.into()),
        Ok(metadata)=>{ensure!(metadata.is_file()&&metadata.len()<=32768,"Invalid MCP configuration file");Ok(Some(serde_json::from_reader(fs::File::open(path)?)?))}
    }
}
fn effective(root:&Path)->Result<(Endpoint,bool)>{
    match read(root)?{
        Some(ep)=>Ok((ep,false)),
        None=>Ok((Endpoint{url:std::env::var("HERMES_RUST_MCP_URL").unwrap_or_default(),token:std::env::var("HERMES_RUST_MCP_TOKEN").ok()},true))
    }
}
pub fn summary(root:&Path)->Result<Summary>{
    let (ep,environment)=effective(root)?;
    let url=if ep.url.is_empty(){String::new()}else{validated(&ep.url)?.to_string()};
    Ok(Summary{url,has_token:ep.token.as_deref().is_some_and(|s|!s.is_empty()),environment})
}
pub(crate) fn validated(url:&str)->Result<Url>{
    let url=Url::parse(url.trim()).context("Invalid MCP server URL")?;
    ensure!(url.scheme()=="https" || (url.scheme()=="http" && matches!(url.host_str(),Some("localhost"|"127.0.0.1"|"[::1]"))),"MCP requires HTTPS or a loopback HTTP endpoint");
    ensure!(url.host().is_some()&&url.username().is_empty()&&url.password().is_none()&&url.query().is_none()&&url.fragment().is_none(),"Enter an MCP URL without embedded credentials, query or fragment");
    Ok(url)
}
pub fn save(root:&Path,url:&str,new_key:&str,clear_key:bool)->Result<Summary>{
    let lock=fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(root.join("mcp-endpoint.lock"))?;
    fs2::FileExt::lock_exclusive(&lock)?;
    let (old,_)=effective(root)?;
    let url=if url.trim().is_empty(){String::new()}else{validated(url)?.to_string()};
    ensure!(url.len()<8192&&new_key.len()<8192&&!new_key.chars().any(char::is_control),"Endpoint URL or credential is invalid");
    let same=validated(&old.url).is_ok_and(|old|old.as_str()==url);
    let token=if clear_key||url.is_empty(){None}else if !new_key.is_empty(){Some(new_key.into())}else if same{old.token}else{None};
    let endpoint=Endpoint{url,token};
    let mut temp=tempfile::NamedTempFile::new_in(root)?;
    serde_json::to_writer_pretty(&mut temp,&endpoint)?;temp.flush()?;temp.as_file().sync_all()?;
    temp.persist(root.join("mcp-endpoint.json"))?;
    summary(root)
}
pub(crate) async fn connect(root:&Path)->Result<Option<Mcp>>{
    let (endpoint,_)=effective(root)?;
    if endpoint.url.trim().is_empty(){return Ok(None)}
    Ok(Some(Mcp::connect(&endpoint.url,endpoint.token).await?))
}
pub async fn discover(root:&Path)->Result<Vec<String>>{
    let mut server=connect(root).await?.context("Configure an MCP server first")?;
    let names=server.specs().iter().filter_map(|s|s["function"]["name"].as_str().map(str::to_owned)).collect();
    server.close().await;Ok(names)
}
