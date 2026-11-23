//! See stock agent/billing_view.py and app/settings/billing.
use super::{Hermes, Message, settings, ui};
use crate::appearance::text;
use iced::{Element,Fill,Task,widget::{column,row,button}};

#[derive(Default)]
pub(super) struct State{
    request:u64,
    loading:bool,
    rows:Option<Result<Vec<(String,String)>,String>>,
}
#[derive(Clone)]
pub(super) enum Action{Refresh,Loaded(u64,Result<Vec<(String,String)>,String>)}

pub(super) fn update(app:&mut Hermes,action:Action)->Task<Message>{
    let state=&mut app.billing;
    match action{
        Action::Refresh if app.portal_connected && !state.loading=>{
            state.loading=true;state.request=state.request.wrapping_add(1);
            let id=state.request;
            return Task::perform(async{hermes_engine::portal::billing_overview().await.map_err(|e|e.to_string())},move|rows|Message::Billing(Action::Loaded(id,rows)));
        }
        Action::Loaded(id,result) if id==state.request=>{state.loading=false;state.rows=app.portal_connected.then_some(result);}
        _=>{}
    }Task::none()
}
impl State{pub fn clear(&mut self){self.request=self.request.wrapping_add(1);self.loading=false;self.rows=None;}}
pub(super) fn view(app:&Hermes)->Element<'_,Message>{
    let state=&app.billing;let p=ui::palette(app.dark);
    let mut body=column![].spacing(15);
    if !app.portal_connected{
        body=body.push(text("Sign in to see your Nous account balance and billing settings.").size(13))
           .push(button(text("Sign in with Nous Portal").size(13)).on_press(Message::PortalStart));
    }else {
        if let Some(result)=&state.rows {match result{
            Ok(rows) if rows.is_empty()=>body=body.push(text("No billing details are available for this account.").size(13)),
            Ok(rows)=>for (label,value) in rows {body=body.push(row![text(label.as_str()).size(13).width(Fill),text(value.as_str()).size(13)]);},
            Err(error)=>body=body.push(text(error.as_str()).size(13)),
        }}
        let label=if state.loading{"Loading…"}else{"Refresh billing details"};
        body=body.push(row![button(text(label).size(13)).on_press_maybe((!state.loading).then_some(Message::Billing(Action::Refresh))),
            button(text("Manage billing in Nous Portal").size(13)).on_press(Message::Link("https://portal.nousresearch.com/billing".into()))].spacing(8))
            .push(text("Only the account details available to your role are shown.").size(12).color(p.muted));
    }
    settings::settings_content(body.into())
}
