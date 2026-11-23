//! Device-local desktop alerts. Only event types emitted by the native runtime
//! are offered in the preferences. Provider prompts and chat text stay private.
use super::{Hermes, Message, ui};
use iced::{Element, Task, widget::{button, column, container, text}};
use serde::{Deserialize,Serialize};
use std::{io::Write,path::Path};

#[derive(Clone, Copy, Debug)]
pub(super) enum Kind { All, Approval, Ready, Failed, Mute }
#[derive(Clone,Debug)]
pub(super) enum Action { Focus(bool), Set(Kind,bool), Test, Receipt(Result<(),String>,bool) }
#[derive(Clone,Serialize,Deserialize)]
#[serde(default)]
struct Prefs {enabled:bool,approval:bool,ready:bool,failed:bool,mute:bool}
impl Default for Prefs {fn default()->Self{Self{enabled:true,approval:true,ready:true,failed:true,mute:false}}}
impl Prefs {
    fn field(&mut self,kind:Kind)->&mut bool {
        match kind{Kind::All=>&mut self.enabled,Kind::Approval=>&mut self.approval,Kind::Ready=>&mut self.ready,Kind::Failed=>&mut self.failed,Kind::Mute=>&mut self.mute}
    }
}
pub(super) struct State {prefs:Prefs, focused:bool, feedback:Option<Result<String,String>>}
impl Default for State{fn default()->Self{Self{prefs:Prefs::default(),focused:true,feedback:None}}}
impl State{
    pub fn load(root:&Path)->Self{
        let mut state=Self::default();
        match std::fs::File::open(root.join("desktop-notifications.json")){
            Ok(file)=>match serde_json::from_reader(file){Ok(p)=>state.prefs=p,Err(e)=>state.feedback=Some(Err(e.to_string()))},
            Err(e) if e.kind()==std::io::ErrorKind::NotFound=>{},
            Err(e)=>state.feedback=Some(Err(e.to_string())),
        } state
    }
}
pub(super) fn focus(event:iced::Event,_:iced::event::Status,_:iced::window::Id)->Option<Message>{
    match event{
        iced::Event::Window(iced::window::Event::Focused)=>Some(Message::Notification(Action::Focus(true))),
        iced::Event::Window(iced::window::Event::Unfocused)=>Some(Message::Notification(Action::Focus(false))),
        _=>None,
    }
}
pub(super) fn on_event(app:&Hermes,event:&Message)->Task<Message>{
    let state=&app.notifications;
    if state.focused || !state.prefs.enabled {return Task::none();}
    let body=match event {
        Message::Engine(id,hermes_engine::Event::Approval{..}) if *id==app.run_id && state.prefs.approval => "Approval needed",
        Message::Finished(id,history,_) if *id==app.run_id => match history.turns.last().map(|turn|&turn.status){
            Some(hermes_engine::Status::Complete) if state.prefs.ready=>"Response ready",
            Some(hermes_engine::Status::Failed(_)) if state.prefs.failed=>"Turn failed",
            _=>return Task::none(),
        },
        _=>return Task::none(),
    };
    send(body,state.prefs.mute,false)
}
fn send(body:&'static str,mute:bool,test:bool)->Task<Message>{
    Task::perform(deliver(body,mute),move|r|Message::Notification(Action::Receipt(r,test)))
}
pub(super) fn update(app:&mut Hermes,action:Action)->Task<Message>{
    let state=&mut app.notifications;
    match action{
        Action::Focus(on)=>state.focused=on,
        Action::Set(kind,on)=>{
            let Some(lib)=&app.library else{return Task::none()};
            let write=||->anyhow::Result<Prefs>{
                let path=lib.root().join("desktop-notifications.json");
                let lock=std::fs::OpenOptions::new().create(true).truncate(false).read(true).write(true).open(path.with_extension("lock"))?;
                fs2::FileExt::lock_exclusive(&lock)?;
                let mut prefs=match std::fs::File::open(&path) { Ok(f)=>serde_json::from_reader(f)?, Err(e) if e.kind()==std::io::ErrorKind::NotFound=>Prefs::default(), Err(e)=>return Err(e.into()) };
                *prefs.field(kind)=on;
                let mut file=tempfile::NamedTempFile::new_in(lib.root())?;
                serde_json::to_writer_pretty(&mut file,&prefs)?; file.flush()?;file.as_file().sync_all()?;file.persist(path)?;
                Ok(prefs)
            };
            match write(){Ok(prefs)=>{state.prefs=prefs;state.feedback=None;},Err(e)=>state.feedback=Some(Err(e.to_string()))}
        }
        Action::Test=>return send("Notifications are working.",state.prefs.mute,true),
        Action::Receipt(Err(e),_)=>state.feedback=Some(Err(e)),
        Action::Receipt(Ok(()),true)=>state.feedback=Some(Ok("Test sent. Check Notification Centre if Focus is enabled.".into())),
        Action::Receipt(Ok(()),false)=>{}
    } Task::none()
}
pub(super) fn view(app:&Hermes)->Element<'_,Message>{
    let p=ui::palette(app.dark);
    let state=&app.notifications;
    let mut body=column![text("OS notifications. Per device.").size(13).color(p.muted)].spacing(16);
    let mut rows=Vec::new();
    for (kind,name,description,on) in [
        (Kind::All,"Enable notifications","Off silences every notification below.",state.prefs.enabled),
        (Kind::Approval,"Approval needed","A command is waiting for you to approve or reject it.",state.prefs.approval),
        (Kind::Ready,"Response ready","A turn finished while Hermes was in the background.",state.prefs.ready),
        (Kind::Failed,"Turn failed","Background turn errors.",state.prefs.failed),
        (Kind::Mute,"Mute notification sounds","Deliver OS notifications silently.",state.prefs.mute),
    ]{
        let enabled=state.prefs.enabled||matches!(kind,Kind::All|Kind::Mute);
        let control:Element<'_,Message>=if enabled {super::settings::switch(on,move|on|Message::Notification(Action::Set(kind,on)),app.dark)}
            else {container(text("Off").size(12).color(p.muted)).into()};
        rows.push(super::settings::card_row(name,description,control,app.dark));
    }
    body=body.push(super::settings::card(rows,app.dark));
    body=body.push(button(text("Send test notification").size(13)).on_press_maybe(state.prefs.enabled.then_some(Message::Notification(Action::Test))));
    if let Some(result)=&state.feedback{body=body.push(match result{Ok(s)=>text(s).color(p.muted),Err(s)=>text(s).color(iced::Color::from_rgb8(194,61,64))}.size(12));}
    super::settings::settings_content(body.into())
}
#[cfg(target_os="linux")]
async fn deliver(body:&str,mute:bool)->Result<(),String>{
    use std::collections::HashMap;
    use zbus::zvariant::Value;
    let connection=zbus::Connection::session().await.map_err(|e|e.to_string())?;
    let proxy=zbus::Proxy::new(&connection,"org.freedesktop.Notifications","/org/freedesktop/Notifications","org.freedesktop.Notifications").await.map_err(|e|e.to_string())?;
    let mut hints=HashMap::new();hints.insert("suppress-sound",Value::from(mute));
    hints.insert("desktop-entry",Value::from("com.nousresearch.hermes.rust-development"));
    let _:u32=proxy.call("Notify",&("Hermes",0u32,"com.nousresearch.hermes.rust-development","Hermes",body,Vec::<String>::new(),hints,-1i32)).await.map_err(|e|e.to_string())?;
    Ok(())
}
#[cfg(target_os="windows")]
async fn deliver(body:&str,mute:bool)->Result<(),String>{
    const APP_ID:&str="NousResearch.Hermes.RustDevelopment";
    let body=body.to_owned();
    tokio::task::spawn_blocking(move||{
        use std::{os::windows::ffi::OsStrExt,ffi::OsStr,ptr};
        use windows_sys::Win32::System::Registry::*;
        fn wide(s:&OsStr)->Vec<u16>{s.encode_wide().chain(Some(0)).collect()}
        let mut key=ptr::null_mut();
        let app=wide(OsStr::new(&format!("Software\\Classes\\AppUserModelId\\{APP_ID}")));
        unsafe{
            let status=RegCreateKeyExW(HKEY_CURRENT_USER,app.as_ptr(),0,ptr::null(),REG_OPTION_NON_VOLATILE,KEY_SET_VALUE,ptr::null(),&mut key,ptr::null_mut());
            if status!=0{return Err(std::io::Error::from_raw_os_error(status as i32).to_string())}
            let name=wide(OsStr::new("DisplayName"));let value=wide(OsStr::new("Hermes"));
            let status=RegSetValueExW(key,name.as_ptr(),0,REG_SZ,value.as_ptr().cast(),(value.len()*2) as u32);
            RegCloseKey(key);
            if status!=0{return Err(std::io::Error::from_raw_os_error(status as i32).to_string())}
        }
        winrt_notification::Toast::new(APP_ID).title("Hermes").text1(&body)
            .sound((!mute).then_some(winrt_notification::Sound::Default)).show().map_err(|e|e.to_string())
    }).await.map_err(|e|e.to_string())?
}
#[cfg(not(any(target_os="linux",target_os="windows",target_os="macos")))]
async fn deliver(_:&str,_:bool)->Result<(),String>{Err("Native alerts are unavailable on this operating system.".into())}
#[cfg(target_os="macos")]
async fn deliver(body:&str,mute:bool)->Result<(),String>{
    use block2::RcBlock;
    use objc2::runtime::Bool;
    use objc2_foundation::{NSBundle,NSError,NSString};
    use objc2_user_notifications::{UNAuthorizationOptions as Options,UNMutableNotificationContent,UNNotificationRequest,UNNotificationSound,UNUserNotificationCenter};
    // Objective-C objects and blocks are created and registered in synchronous
    // scopes. Only the owned Rust channels cross suspension points.
    let (send,auth)=async_channel::bounded(1);
    unsafe{
        if NSBundle::mainBundle().bundleIdentifier().is_none(){return Err("Open the packaged Hermes app to send macOS notifications.".into());}
        let callback=RcBlock::new(move|granted:Bool,error:*mut NSError|{
            let outcome=if let Some(err)=error.as_ref(){Err(err.localizedDescription().to_string())}else if granted.as_bool(){Ok(())}else{Err("Allow Hermes notifications in macOS System Settings.".into())};
            let _=send.try_send(outcome);
        });
        UNUserNotificationCenter::currentNotificationCenter().requestAuthorizationWithOptions_completionHandler(Options::UNAuthorizationOptionAlert|Options::UNAuthorizationOptionSound,&callback);
    }
    auth.recv().await.map_err(|_|"Notification authorization was interrupted")??;
    let (send,delivered)=async_channel::bounded(1);
    unsafe{
        let content=UNMutableNotificationContent::new();
        content.setTitle(&NSString::from_str("Hermes"));content.setBody(&NSString::from_str(body));
        if !mute{content.setSound(Some(&UNNotificationSound::defaultSound()));}
        static NEXT:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
        let id=NSString::from_str(&format!("hermes.{}.{}",std::process::id(),NEXT.fetch_add(1,std::sync::atomic::Ordering::Relaxed)));
        let request=UNNotificationRequest::requestWithIdentifier_content_trigger(&id,&content,None);
        let callback=RcBlock::new(move|error:*mut NSError|{let _=send.try_send(error.as_ref().map_or(Ok(()),|e|Err(e.localizedDescription().to_string())));});
        UNUserNotificationCenter::currentNotificationCenter().addNotificationRequest_withCompletionHandler(&request,Some(&callback));
    }
    delivered.recv().await.map_err(|_|"macOS did not acknowledge the notification")?
}
