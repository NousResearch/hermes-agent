//! Port of composer/reasoning-pill.tsx and shell/model-edit-submenu.tsx.
use super::{Hermes,Message,Screen,appearance::text,ui};
use iced::{Alignment,Border,Element,Fill,Rectangle,Task,advanced::widget::{self,Id,Operation,operation::Outcome},keyboard,widget::{Space,button,column,container,mouse_area,opaque,pin,row,rule,stack,toggler}};
pub(super) const TRIGGER:&str="composer.reasoning";
pub(super) const LEVELS:[(&str,&str);7]=[("minimal","Minimal"),("low","Low"),("medium","Medium"),("high","High"),("xhigh","Extra High"),("max","Max"),("ultra","Ultra")];

#[derive(Clone)]
pub(super) enum Action{Open,Anchor(String,Option<Rectangle>),Close,Pick(&'static str),Toggle,Fast}
#[derive(Default)]
pub(super) struct State{pub anchor:Option<Rectangle>,cursor:usize}
pub(super) fn effective(app:&Hermes)->&str{app.models.options.reasoning_effort.as_deref().unwrap_or("medium")}
pub(super) fn short(effort:Option<&str>)->&'static str{
    match effort.unwrap_or("medium"){"none"=>"Off","minimal"=>"Min","low"=>"Low","medium"=>"Med","high"=>"High","xhigh"=>"XHigh","max"=>"Max","ultra"=>"Ultra→Max",_=>"Med"}
}
fn supports(app:&Hermes)->bool{app.models.current_model(&app.config).is_some_and(|m|m.supports("reasoning")||m.supports("reasoning_effort"))}
fn fast(app:&Hermes)->bool{app.models.current_model(&app.config).is_some_and(|m|m.supports("service_tier"))}
fn allows_off(app:&Hermes)->bool{!app.config.model.rsplit('/').next().unwrap_or_default().starts_with("gpt-6.1-sol")}
fn key(app:&Hermes)->String{format!("{}\n{}",app.config.base_url,app.config.model)}
struct FindTrigger(Option<Rectangle>);
impl Operation<Option<Rectangle>> for FindTrigger{
    fn traverse(&mut self,f:&mut dyn FnMut(&mut dyn Operation<Option<Rectangle>>)){f(self)}
    fn container(&mut self,id:Option<&Id>,bounds:Rectangle){if id==Some(&Id::new(TRIGGER)){self.0=Some(bounds);}}
    fn finish(&self)->Outcome<Option<Rectangle>>{Outcome::Some(self.0)}
}
pub(super) fn update(app:&mut Hermes,action:Action)->Task<Message>{
    match action{
        Action::Open if !app.running && app.screen==Screen::Chat && supports(app)=>{
            if app.reasoning.anchor.is_some(){return update(app,Action::Close);}
            let key=key(app);
            return widget::operate(FindTrigger(None)).map(move|bounds|Message::Reasoning(Action::Anchor(key.clone(),bounds)));
        }
        Action::Anchor(expected,bounds) if expected==key(app) && app.screen==Screen::Chat=>{
            app.reasoning.anchor=bounds;app.reasoning.cursor=LEVELS.iter().position(|(v,_)|*v==effective(app)).unwrap_or(2);
        }
        Action::Close=>{app.reasoning.anchor=None;return iced::widget::operation::focus(super::COMPOSER_ID);}
        Action::Toggle if allows_off(app)=>{
            let value=if effective(app)=="none"{"medium"}else{"none"};
            return update(app,Action::Pick(value));
        }
        action @ (Action::Pick(_)|Action::Fast) if supports(app) && !app.running && app.reasoning.anchor.is_some()=>{
            let old=app.models.options.clone();
            if let Action::Pick(level)=action{
                if level!="none"&&!LEVELS.iter().any(|(v,_)|*v==level){return Task::none();}
                app.models.options.reasoning_effort=Some(level.into());
                app.models.options.reasoning_object=app.models.current_model(&app.config).is_some_and(|m|m.supports("reasoning"));
                app.reasoning.cursor=LEVELS.iter().position(|(v,_)|*v==level).unwrap_or(2);
            }else if fast(app){app.models.options.fast=!old.fast;}else{return Task::none();}
            if let Err(error)=app.models.save_options(&app.config){
                app.models.options=old;app.error(error);
            }else{app.models.reasoning=app.models.options.reasoning_effort.clone().unwrap_or_default();}
        }
        _=>{}
    }Task::none()
}
pub(super) fn keyboard(app:&mut Hermes,event:&keyboard::Event)->Option<Task<Message>>{
    if app.reasoning.anchor.is_none(){return None}
    if let keyboard::Event::KeyPressed{key:keyboard::Key::Named(k),..}=event{
        use keyboard::key::Named;
        match k{
            Named::Escape|Named::Tab=>return Some(update(app,Action::Close)),
            Named::ArrowDown=>app.reasoning.cursor=(app.reasoning.cursor+1)%LEVELS.len(),
            Named::ArrowUp=>app.reasoning.cursor=(app.reasoning.cursor+LEVELS.len()-1)%LEVELS.len(),
            Named::Home=>app.reasoning.cursor=0,Named::End=>app.reasoning.cursor=LEVELS.len()-1,
            Named::Enter|Named::Space=>return Some(update(app,Action::Pick(LEVELS[app.reasoning.cursor].0))),
            _=>{}
        }
    }Some(Task::none())
}
/// `below`: open under the trigger (the HUD's composer sits at the top of its band).
pub(super) fn overlay(app:&Hermes,bounds:Rectangle,below:bool)->Element<'_,Message>{
    let p=ui::palette(app.dark);let picked=effective(app);
    let mut items=column![text("Options").size(11).color(p.muted)].padding(6).spacing(1);
    if allows_off(app){
        items=items.push(row![text("Thinking").size(13).width(Fill),toggler(picked!="none").size(17).on_toggle(|_|Message::Reasoning(Action::Toggle))].align_y(Alignment::Center).height(27).padding([0,5]));
    }
    if fast(app){
        items=items.push(row![text("Fast").size(13).width(Fill),toggler(app.models.options.fast).size(17).on_toggle(|_|Message::Reasoning(Action::Fast))].align_y(Alignment::Center).height(27).padding([0,5]));
    }
    items=items.push(rule::horizontal(1)).push(text("Effort").size(11).color(p.muted));
    for (index,(id,label)) in LEVELS.iter().enumerate(){
        let selected=picked==*id;let hovered=index==app.reasoning.cursor;
        let title=if selected && *id=="ultra"{"Ultra (sends Max on this route)"}else{label};
        let item=row![text(if selected{"●"}else{""}).size(9).width(13),text(title).size(13),Space::new().width(Fill)].align_y(Alignment::Center);
        items=items.push(button(item).height(26).width(Fill).padding([2,5]).on_press(Message::Reasoning(Action::Pick(id))).style(move|_,status|{
            button::Style{background:(matches!(status,button::Status::Hovered|button::Status::Pressed)||hovered).then_some(p.secondary.into()),text_color:p.foreground,border:Border{radius:3.into(),..Default::default()},..Default::default()}
        }));
    }
    let height=12.+14.+(if allows_off(app){28.}else{0.})+(if fast(app){28.}else{0.})+1.+15.+7.*27.+5.;
    let panel=container(items).width(208).style(move |_|container::Style{background:Some(p.card.into()),text_color:Some(p.foreground),border:Border{radius:6.into(),width:1.,color:p.border},..Default::default()});
    stack![
        mouse_area(Space::new().width(Fill).height(Fill)).on_press(Message::Reasoning(Action::Close)),
        pin(opaque(panel)).x((bounds.x+bounds.width-208.).max(4.)).y(if below {bounds.y+bounds.height+6.} else {(bounds.y-height-8.).max(4.)})
    ].width(Fill).height(Fill).into()
}
