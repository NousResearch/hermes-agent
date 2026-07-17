var wn=Object.defineProperty;var vn=(t,e,r)=>e in t?wn(t,e,{enumerable:!0,configurable:!0,writable:!0,value:r}):t[e]=r;var ct=(t,e,r)=>vn(t,typeof e!="symbol"?e+"":e,r);import{host as Tu,ROUTES_AREA as Su,SIDEBAR_NAV_AREA as Ru}from"@hermes/plugin-sdk";import{useCallback as Iu,useEffect as Nu,useRef as mn,useState as Ve}from"react";var Bu=8*1024;var Du=8*1024;var _n=32*1024;var Ye=class extends Error{constructor(e="boardstate ws transport is closed"){super(e);ct(this,"code","transport_closed");this.name="WsTransportClosedError"}},xn=class extends Error{constructor(e="no WebSocket implementation available (need a browser, Node \u2265 22, or WebSocketImpl)"){super(e);ct(this,"code","transport_unavailable");this.name="WsTransportUnavailableError"}};function $n(t){if(t)return t;let e=globalThis.WebSocket;if(e)return e;throw new xn}function Ze(t,e={}){let r=new Map,s=new Map,n=[],o=null,a=1,i=!1,l=!1,u=()=>{},b=()=>{},h=new Promise((g,_)=>{u=g,b=_});h.catch(()=>{});function m(g){for(let _ of r.values())_.reject(g);r.clear(),n.length=0}function w(){if(i)return;i=!0;let g=new Ye;b(g),m(g);try{o?.close()}catch{}}function v(g){if(typeof g!="string")return;let _;try{_=JSON.parse(g)}catch{return}if(typeof _!="object"||_===null)return;let $=_;if(typeof $.id=="number"){let D=r.get($.id);if(!D)return;if(r.delete($.id),$.error){let I=typeof $.error.message=="string"?$.error.message:"boardstate error",Y=new Error(I);typeof $.error.code=="string"&&(Y.code=$.error.code),D.reject(Y)}else D.resolve($.result);return}if(typeof $.event=="string"){let D=s.get($.event);if(!D)return;for(let I of[...D])I($.payload)}}try{let g=new($n(e.WebSocketImpl))(t);o=g,g.addEventListener("open",()=>{l=!0,u();for(let _ of n)g.send(_);n.length=0}),g.addEventListener("message",_=>v(_.data)),g.addEventListener("error",()=>w()),g.addEventListener("close",()=>w())}catch(g){i=!0,b(g),m(g)}return{get closed(){return i},ready:h,request(g,_,$){if(i)return Promise.reject(new Ye);let D=a++,I=JSON.stringify({id:D,method:g,params:_??{}});return new Promise((Y,Wt)=>{r.set(D,{resolve:Y,reject:Wt}),l&&o?o.send(I):n.push(I)})},addEventListener(g,_){let $=s.get(g);return $||($=new Set,s.set(g,$)),$.add(_),()=>{$?.delete(_)}},close(){w()}}}function U(t,e,r,s){return{x:t,y:e,w:r,h:s}}var An=[{kind:"builtin:stat-card",summary:"One number that matters \u2014 a KPI with a label.",bindings:[{key:"value",shape:"number | string, or a structured payload + props.metric"}],props:{format:'"usd" | "int" | "percent" | "raw" (how the number renders)',metric:"when the binding resolves an object, the field name to display",label:"inner label (omit if it would just repeat the title)"},example:{id:"mrr",kind:"builtin:stat-card",title:"MRR",grid:U(0,0,3,2),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:128400}},props:{format:"usd",label:"Monthly recurring revenue"}}},{kind:"builtin:chart",summary:"Trends, comparisons, budgets \u2014 a small inline chart.",bindings:[{key:"value",shape:"number[] (or labeled points {label,value}[])"}],props:{type:'"line" | "bar" | "area" | "sparkline" | "gauge" (default line)',detail:"true adds labeled axes, gridlines, and value tooltips (line/bar/area)",label:"sparkline only: true shows the trailing value as an end label"},example:{id:"revenue-trend",kind:"builtin:chart",title:"Revenue (14d)",grid:U(0,2,8,5),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:[8,12,10,18,24,21,30,35,41,52]}},props:{type:"area"}},examples:[{id:"signups-spark",kind:"builtin:chart",title:"Signups",grid:U(0,7,3,2),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:[12,9,14,11,17,15,22]}},props:{type:"sparkline",label:!0}},{id:"latency-detail",kind:"builtin:chart",title:"p95 latency (ms)",grid:U(0,9,8,5),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:[180,220,190,240,210,260,230]}},props:{type:"line",detail:!0}}]},{kind:"builtin:table",summary:"Rows and columns \u2014 a compact table (keep ~10 visible rows).",bindings:[{key:"rows",shape:"Array<Record<string, unknown>> \u2014 NOT `value`"}],props:{columns:"string[] of keys to show (defaults to the first row's keys)",limit:"max visible rows before a \u201C+N more\u201D count"},example:{id:"recent-runs",kind:"builtin:table",title:"Recent runs",grid:U(0,7,8,4),collapsed:!1,hidden:!1,bindings:{rows:{source:"static",value:[{agent:"finance",task:"Q3 rollup",status:"done"},{agent:"ops",task:"Log sweep",status:"running"}]}},props:{columns:["agent","task","status"]}}},{kind:"builtin:markdown",summary:"Prose, explanations, small markdown tables (sanitized).",bindings:[{key:"content",shape:"markdown string \u2014 NOT `value`"}],props:{markdown:"inline markdown source (used when there is no `content` binding)",text:"alias for `markdown`"},example:{id:"summary",kind:"builtin:markdown",title:"Summary",grid:U(8,2,4,5),collapsed:!1,hidden:!1,props:{markdown:`## Insights

- Signal up **6.5\xD7** across 14 days.
- Momentum late.`}}},{kind:"builtin:notes",summary:"Operator scratch text (persisted via widget state).",bindings:[],props:{text:"starter content"},example:{id:"scratchpad",kind:"builtin:notes",title:"Notes",grid:U(8,7,4,4),collapsed:!1,hidden:!1,props:{text:"Jot findings here\u2026"}}},{kind:"builtin:activity",summary:"An event feed \u2014 recent things that happened.",bindings:[{key:"value",shape:"{ entries: [{ ts, jobName, status, summary }] }"}],props:{limit:"max entries shown"},example:{id:"agent-events",kind:"builtin:activity",title:"Agent events",grid:U(0,11,6,4),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:{entries:[{ts:17836e8,jobName:"finance",status:"ok",summary:"Rollup posted"}]}}}}},{kind:"builtin:action-form",summary:"The chat\u2194dashboard loop \u2014 a form that submits through the control plane.",bindings:[],props:{template:"the message sent on submit; `{{fieldName}}` interpolates a field (single pass)",fields:'array of { name, label, type: "text"|"number"|"select", options?, maxLength? }',buttonLabel:"the submit button text (optional)",mode:'"prompt" (default: submit the template to the agent) or "tool" (invoke a granted external tool)',connector:"tool mode only: the granted connector name (SPEC \xA717 v2)",tool:"tool mode only: the tool to invoke on that connector",argsFrom:"tool mode only: map of tool-arg name \u2192 declared field name"},example:{id:"ask-agent",kind:"builtin:action-form",title:"Ask the agent",grid:U(0,0,4,3),collapsed:!1,hidden:!1,props:{template:"Summarize {{topic}} for the board.",fields:[{name:"topic",label:"Topic",type:"text"}],buttonLabel:"Ask"}},examples:[{id:"file-ticket",kind:"builtin:action-form",title:"File a ticket",grid:U(0,0,4,4),collapsed:!1,hidden:!1,props:{mode:"tool",connector:"linear",tool:"create_issue",template:"Create issue: {title}",fields:[{name:"title",label:"Title",type:"text",maxLength:120},{name:"priority",label:"Priority",type:"select",options:["low","med","high"]}],argsFrom:{title:"title",priority:"priority"},buttonLabel:"Create"}}]},{kind:"builtin:action-button",summary:"One click \u2192 invoke a granted external tool with fixed args (operator-confirmed).",bindings:[],props:{connector:"the granted connector name (SPEC \xA717 v2)",tool:"the tool to invoke on that connector",args:"fixed argument object passed on click (optional)",label:"button text (optional)"},example:{id:"restart-worker",kind:"builtin:action-button",title:"Restart worker",grid:U(0,0,3,2),collapsed:!1,hidden:!1,props:{connector:"officecli",tool:"restart_service",args:{service:"worker"},label:"Restart"}}},{kind:"builtin:chat",summary:"Talk to the agent and watch it work (ignores bindings).",bindings:[],props:{placeholder:"empty-input hint text"},example:{id:"assistant",kind:"builtin:chat",title:"Assistant",grid:U(0,0,6,8),collapsed:!1,hidden:!1,props:{placeholder:"Ask me to build a view\u2026"}}}],kn=[{kind:"builtin:sessions",summary:"Who/what is running.",valueShape:"rows { key, label, status, hasActiveRun, updatedAt }; props.limit"},{kind:"builtin:agent-status",summary:"Agents + goals/progress.",valueShape:"sessions shape + goal { objective, tokensUsed, tokenBudget }"},{kind:"builtin:usage",summary:"Cost/token totals.",valueShape:"{ totals: { totalCost, totalTokens }, days? }"},{kind:"builtin:cron",summary:"Scheduled jobs.",valueShape:"{ jobs: [{ id, name, enabled, state: { nextRunAtMs, lastRunStatus } }] }"},{kind:"builtin:instances",summary:"Fleet presence.",valueShape:"{ presence: [{ instanceId, platform, version, lastInputSeconds }] }"},{kind:"builtin:approvals",summary:"Pending widget approvals (reads the live registry; ignores bindings).",valueShape:"none \u2014 reads the registry"},{kind:"builtin:preview",summary:"A live page preview.",valueShape:"props.url (same-origin ok; cross-origin needs host opt-in)"},{kind:"builtin:iframe-embed",summary:"An embedded live page.",valueShape:"props.url (same-origin ok; cross-origin needs host opt-in)"}],Zu=[...An.map(t=>t.kind),...kn.map(t=>t.kind)];var Qu=256*1024;var tb=64*1024;var eb=512*1024,rb=512*1024,sb=256*1024;var zt=globalThis,Re=zt.ShadowRoot&&(zt.ShadyCSS===void 0||zt.ShadyCSS.nativeShadow)&&"adoptedStyleSheets"in Document.prototype&&"replace"in CSSStyleSheet.prototype,Zr=Symbol(),Qe=new WeakMap,En=class{constructor(t,e,r){if(this._$cssResult$=!0,r!==Zr)throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");this.cssText=t,this.t=e}get styleSheet(){let t=this.o,e=this.t;if(Re&&t===void 0){let r=e!==void 0&&e.length===1;r&&(t=Qe.get(e)),t===void 0&&((this.o=t=new CSSStyleSheet).replaceSync(this.cssText),r&&Qe.set(e,t))}return t}toString(){return this.cssText}},Tn=t=>new En(typeof t=="string"?t:t+"",void 0,Zr),Sn=(t,e)=>{if(Re)t.adoptedStyleSheets=e.map(r=>r instanceof CSSStyleSheet?r:r.styleSheet);else for(let r of e){let s=document.createElement("style"),n=zt.litNonce;n!==void 0&&s.setAttribute("nonce",n),s.textContent=r.cssText,t.appendChild(s)}},tr=Re?t=>t:t=>t instanceof CSSStyleSheet?(e=>{let r="";for(let s of e.cssRules)r+=s.cssText;return Tn(r)})(t):t;var{is:Rn,defineProperty:In,getOwnPropertyDescriptor:Nn,getOwnPropertyNames:Mn,getOwnPropertySymbols:Cn,getPrototypeOf:On}=Object,Q=globalThis,er=Q.trustedTypes,Bn=er?er.emptyScript:"",Dn=Q.reactiveElementPolyfillSupport,kt=(t,e)=>t,ge={toAttribute(t,e){switch(e){case Boolean:t=t?Bn:null;break;case Object:case Array:t=t==null?t:JSON.stringify(t)}return t},fromAttribute(t,e){let r=t;switch(e){case Boolean:r=t!==null;break;case Number:r=t===null?null:Number(t);break;case Object:case Array:try{r=JSON.parse(t)}catch{r=null}}return r}},Qr=(t,e)=>!Rn(t,e),rr={attribute:!0,type:String,converter:ge,reflect:!1,useDefault:!1,hasChanged:Qr};Symbol.metadata??(Symbol.metadata=Symbol("metadata")),Q.litPropertyMetadata??(Q.litPropertyMetadata=new WeakMap);var ht=class extends HTMLElement{static addInitializer(t){this._$Ei(),(this.l??(this.l=[])).push(t)}static get observedAttributes(){return this.finalize(),this._$Eh&&[...this._$Eh.keys()]}static createProperty(t,e=rr){if(e.state&&(e.attribute=!1),this._$Ei(),this.prototype.hasOwnProperty(t)&&((e=Object.create(e)).wrapped=!0),this.elementProperties.set(t,e),!e.noAccessor){let r=Symbol(),s=this.getPropertyDescriptor(t,r,e);s!==void 0&&In(this.prototype,t,s)}}static getPropertyDescriptor(t,e,r){let{get:s,set:n}=Nn(this.prototype,t)??{get(){return this[e]},set(o){this[e]=o}};return{get:s,set(o){let a=s?.call(this);n?.call(this,o),this.requestUpdate(t,a,r)},configurable:!0,enumerable:!0}}static getPropertyOptions(t){return this.elementProperties.get(t)??rr}static _$Ei(){if(this.hasOwnProperty(kt("elementProperties")))return;let t=On(this);t.finalize(),t.l!==void 0&&(this.l=[...t.l]),this.elementProperties=new Map(t.elementProperties)}static finalize(){if(this.hasOwnProperty(kt("finalized")))return;if(this.finalized=!0,this._$Ei(),this.hasOwnProperty(kt("properties"))){let e=this.properties,r=[...Mn(e),...Cn(e)];for(let s of r)this.createProperty(s,e[s])}let t=this[Symbol.metadata];if(t!==null){let e=litPropertyMetadata.get(t);if(e!==void 0)for(let[r,s]of e)this.elementProperties.set(r,s)}this._$Eh=new Map;for(let[e,r]of this.elementProperties){let s=this._$Eu(e,r);s!==void 0&&this._$Eh.set(s,e)}this.elementStyles=this.finalizeStyles(this.styles)}static finalizeStyles(t){let e=[];if(Array.isArray(t)){let r=new Set(t.flat(1/0).reverse());for(let s of r)e.unshift(tr(s))}else t!==void 0&&e.push(tr(t));return e}static _$Eu(t,e){let r=e.attribute;return r===!1?void 0:typeof r=="string"?r:typeof t=="string"?t.toLowerCase():void 0}constructor(){super(),this._$Ep=void 0,this.isUpdatePending=!1,this.hasUpdated=!1,this._$Em=null,this._$Ev()}_$Ev(){this._$ES=new Promise(t=>this.enableUpdating=t),this._$AL=new Map,this._$E_(),this.requestUpdate(),this.constructor.l?.forEach(t=>t(this))}addController(t){(this._$EO??(this._$EO=new Set)).add(t),this.renderRoot!==void 0&&this.isConnected&&t.hostConnected?.()}removeController(t){this._$EO?.delete(t)}_$E_(){let t=new Map,e=this.constructor.elementProperties;for(let r of e.keys())this.hasOwnProperty(r)&&(t.set(r,this[r]),delete this[r]);t.size>0&&(this._$Ep=t)}createRenderRoot(){let t=this.shadowRoot??this.attachShadow(this.constructor.shadowRootOptions);return Sn(t,this.constructor.elementStyles),t}connectedCallback(){this.renderRoot??(this.renderRoot=this.createRenderRoot()),this.enableUpdating(!0),this._$EO?.forEach(t=>t.hostConnected?.())}enableUpdating(t){}disconnectedCallback(){this._$EO?.forEach(t=>t.hostDisconnected?.())}attributeChangedCallback(t,e,r){this._$AK(t,r)}_$ET(t,e){let r=this.constructor.elementProperties.get(t),s=this.constructor._$Eu(t,r);if(s!==void 0&&r.reflect===!0){let n=(r.converter?.toAttribute!==void 0?r.converter:ge).toAttribute(e,r.type);this._$Em=t,n==null?this.removeAttribute(s):this.setAttribute(s,n),this._$Em=null}}_$AK(t,e){let r=this.constructor,s=r._$Eh.get(t);if(s!==void 0&&this._$Em!==s){let n=r.getPropertyOptions(s),o=typeof n.converter=="function"?{fromAttribute:n.converter}:n.converter?.fromAttribute!==void 0?n.converter:ge;this._$Em=s;let a=o.fromAttribute(e,n.type);this[s]=a??this._$Ej?.get(s)??a,this._$Em=null}}requestUpdate(t,e,r,s=!1,n){if(t!==void 0){let o=this.constructor;if(s===!1&&(n=this[t]),r??(r=o.getPropertyOptions(t)),!((r.hasChanged??Qr)(n,e)||r.useDefault&&r.reflect&&n===this._$Ej?.get(t)&&!this.hasAttribute(o._$Eu(t,r))))return;this.C(t,e,r)}this.isUpdatePending===!1&&(this._$ES=this._$EP())}C(t,e,{useDefault:r,reflect:s,wrapped:n},o){r&&!(this._$Ej??(this._$Ej=new Map)).has(t)&&(this._$Ej.set(t,o??e??this[t]),n!==!0||o!==void 0)||(this._$AL.has(t)||(this.hasUpdated||r||(e=void 0),this._$AL.set(t,e)),s===!0&&this._$Em!==t&&(this._$Eq??(this._$Eq=new Set)).add(t))}async _$EP(){this.isUpdatePending=!0;try{await this._$ES}catch(e){Promise.reject(e)}let t=this.scheduleUpdate();return t!=null&&await t,!this.isUpdatePending}scheduleUpdate(){return this.performUpdate()}performUpdate(){if(!this.isUpdatePending)return;if(!this.hasUpdated){if(this.renderRoot??(this.renderRoot=this.createRenderRoot()),this._$Ep){for(let[s,n]of this._$Ep)this[s]=n;this._$Ep=void 0}let r=this.constructor.elementProperties;if(r.size>0)for(let[s,n]of r){let{wrapped:o}=n,a=this[s];o!==!0||this._$AL.has(s)||a===void 0||this.C(s,void 0,n,a)}}let t=!1,e=this._$AL;try{t=this.shouldUpdate(e),t?(this.willUpdate(e),this._$EO?.forEach(r=>r.hostUpdate?.()),this.update(e)):this._$EM()}catch(r){throw t=!1,this._$EM(),r}t&&this._$AE(e)}willUpdate(t){}_$AE(t){this._$EO?.forEach(e=>e.hostUpdated?.()),this.hasUpdated||(this.hasUpdated=!0,this.firstUpdated(t)),this.updated(t)}_$EM(){this._$AL=new Map,this.isUpdatePending=!1}get updateComplete(){return this.getUpdateComplete()}getUpdateComplete(){return this._$ES}shouldUpdate(t){return!0}update(t){this._$Eq&&(this._$Eq=this._$Eq.forEach(e=>this._$ET(e,this[e]))),this._$EM()}updated(t){}firstUpdated(t){}};ht.elementStyles=[],ht.shadowRootOptions={mode:"open"},ht[kt("elementProperties")]=new Map,ht[kt("finalized")]=new Map,Dn?.({ReactiveElement:ht}),(Q.reactiveElementVersions??(Q.reactiveElementVersions=[])).push("2.1.2");var Et=globalThis,sr=t=>t,Ft=Et.trustedTypes,nr=Ft?Ft.createPolicy("lit-html",{createHTML:t=>t}):void 0,Ie="$lit$",K=`lit$${Math.random().toFixed(9).slice(2)}$`,Ne="?"+K,Ln=`<${Ne}>`,nt=document,Nt=()=>nt.createComment(""),Mt=t=>t===null||typeof t!="object"&&typeof t!="function",Me=Array.isArray,ts=t=>Me(t)||typeof t?.[Symbol.iterator]=="function",be=`[ 	
\f\r]`,xt=/<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g,or=/-->/g,ar=/>/g,rt=RegExp(`>|${be}(?:([^\\s"'>=/]+)(${be}*=${be}*(?:[^ 	
\f\r"'\`<>=]|("|')|))|$)`,"g"),ir=/'/g,dr=/"/g,es=/^(?:script|style|textarea|title)$/i,rs=t=>(e,...r)=>({_$litType$:t,strings:e,values:r}),c=rs(1),x=rs(2),ot=Symbol.for("lit-noChange"),p=Symbol.for("lit-nothing"),lr=new WeakMap,st=nt.createTreeWalker(nt,129);function ss(t,e){if(!Me(t)||!t.hasOwnProperty("raw"))throw Error("invalid template strings array");return nr!==void 0?nr.createHTML(e):e}var ns=(t,e)=>{let r=t.length-1,s=[],n,o=e===2?"<svg>":e===3?"<math>":"",a=xt;for(let i=0;i<r;i++){let l=t[i],u,b,h=-1,m=0;for(;m<l.length&&(a.lastIndex=m,b=a.exec(l),b!==null);)m=a.lastIndex,a===xt?b[1]==="!--"?a=or:b[1]!==void 0?a=ar:b[2]!==void 0?(es.test(b[2])&&(n=RegExp("</"+b[2],"g")),a=rt):b[3]!==void 0&&(a=rt):a===rt?b[0]===">"?(a=n??xt,h=-1):b[1]===void 0?h=-2:(h=a.lastIndex-b[2].length,u=b[1],a=b[3]===void 0?rt:b[3]==='"'?dr:ir):a===dr||a===ir?a=rt:a===or||a===ar?a=xt:(a=rt,n=void 0);let w=a===rt&&t[i+1].startsWith("/>")?" ":"";o+=a===xt?l+Ln:h>=0?(s.push(u),l.slice(0,h)+Ie+l.slice(h)+K+w):l+K+(h===-2?i:w)}return[ss(t,o+(t[r]||"<?>")+(e===2?"</svg>":e===3?"</math>":"")),s]},fe=class os{constructor({strings:e,_$litType$:r},s){let n;this.parts=[];let o=0,a=0,i=e.length-1,l=this.parts,[u,b]=ns(e,r);if(this.el=os.createElement(u,s),st.currentNode=this.el.content,r===2||r===3){let h=this.el.content.firstChild;h.replaceWith(...h.childNodes)}for(;(n=st.nextNode())!==null&&l.length<i;){if(n.nodeType===1){if(n.hasAttributes())for(let h of n.getAttributeNames())if(h.endsWith(Ie)){let m=b[a++],w=n.getAttribute(h).split(K),v=/([.?@])?(.*)/.exec(m);l.push({type:1,index:o,name:v[2],strings:w,ctor:v[1]==="."?ds:v[1]==="?"?ls:v[1]==="@"?cs:Lt}),n.removeAttribute(h)}else h.startsWith(K)&&(l.push({type:6,index:o}),n.removeAttribute(h));if(es.test(n.tagName)){let h=n.textContent.split(K),m=h.length-1;if(m>0){n.textContent=Ft?Ft.emptyScript:"";for(let w=0;w<m;w++)n.append(h[w],Nt()),st.nextNode(),l.push({type:2,index:++o});n.append(h[m],Nt())}}}else if(n.nodeType===8)if(n.data===Ne)l.push({type:2,index:o});else{let h=-1;for(;(h=n.data.indexOf(K,h+1))!==-1;)l.push({type:7,index:o}),h+=K.length-1}o++}}static createElement(e,r){let s=nt.createElement("template");return s.innerHTML=e,s}};function at(t,e,r=t,s){if(e===ot)return e;let n=s!==void 0?r._$Co?.[s]:r._$Cl,o=Mt(e)?void 0:e._$litDirective$;return n?.constructor!==o&&(n?._$AO?.(!1),o===void 0?n=void 0:(n=new o(t),n._$AT(t,r,s)),s!==void 0?(r._$Co??(r._$Co=[]))[s]=n:r._$Cl=n),n!==void 0&&(e=at(t,n._$AS(t,e.values),n,s)),e}var as=class{constructor(t,e){this._$AV=[],this._$AN=void 0,this._$AD=t,this._$AM=e}get parentNode(){return this._$AM.parentNode}get _$AU(){return this._$AM._$AU}u(t){let{el:{content:e},parts:r}=this._$AD,s=(t?.creationScope??nt).importNode(e,!0);st.currentNode=s;let n=st.nextNode(),o=0,a=0,i=r[0];for(;i!==void 0;){if(o===i.index){let l;i.type===2?l=new ee(n,n.nextSibling,this,t):i.type===1?l=new i.ctor(n,i.name,i.strings,this,t):i.type===6&&(l=new us(n,this,t)),this._$AV.push(l),i=r[++a]}o!==i?.index&&(n=st.nextNode(),o++)}return st.currentNode=nt,s}p(t){let e=0;for(let r of this._$AV)r!==void 0&&(r.strings!==void 0?(r._$AI(t,r,e),e+=r.strings.length-2):r._$AI(t[e])),e++}},ee=class is{get _$AU(){return this._$AM?._$AU??this._$Cv}constructor(e,r,s,n){this.type=2,this._$AH=p,this._$AN=void 0,this._$AA=e,this._$AB=r,this._$AM=s,this.options=n,this._$Cv=n?.isConnected??!0}get parentNode(){let e=this._$AA.parentNode,r=this._$AM;return r!==void 0&&e?.nodeType===11&&(e=r.parentNode),e}get startNode(){return this._$AA}get endNode(){return this._$AB}_$AI(e,r=this){e=at(this,e,r),Mt(e)?e===p||e==null||e===""?(this._$AH!==p&&this._$AR(),this._$AH=p):e!==this._$AH&&e!==ot&&this._(e):e._$litType$!==void 0?this.$(e):e.nodeType!==void 0?this.T(e):ts(e)?this.k(e):this._(e)}O(e){return this._$AA.parentNode.insertBefore(e,this._$AB)}T(e){this._$AH!==e&&(this._$AR(),this._$AH=this.O(e))}_(e){this._$AH!==p&&Mt(this._$AH)?this._$AA.nextSibling.data=e:this.T(nt.createTextNode(e)),this._$AH=e}$(e){let{values:r,_$litType$:s}=e,n=typeof s=="number"?this._$AC(e):(s.el===void 0&&(s.el=fe.createElement(ss(s.h,s.h[0]),this.options)),s);if(this._$AH?._$AD===n)this._$AH.p(r);else{let o=new as(n,this),a=o.u(this.options);o.p(r),this.T(a),this._$AH=o}}_$AC(e){let r=lr.get(e.strings);return r===void 0&&lr.set(e.strings,r=new fe(e)),r}k(e){Me(this._$AH)||(this._$AH=[],this._$AR());let r=this._$AH,s,n=0;for(let o of e)n===r.length?r.push(s=new is(this.O(Nt()),this.O(Nt()),this,this.options)):s=r[n],s._$AI(o),n++;n<r.length&&(this._$AR(s&&s._$AB.nextSibling,n),r.length=n)}_$AR(e=this._$AA.nextSibling,r){for(this._$AP?.(!1,!0,r);e!==this._$AB;){let s=sr(e).nextSibling;sr(e).remove(),e=s}}setConnected(e){this._$AM===void 0&&(this._$Cv=e,this._$AP?.(e))}},Lt=class{get tagName(){return this.element.tagName}get _$AU(){return this._$AM._$AU}constructor(t,e,r,s,n){this.type=1,this._$AH=p,this._$AN=void 0,this.element=t,this.name=e,this._$AM=s,this.options=n,r.length>2||r[0]!==""||r[1]!==""?(this._$AH=Array(r.length-1).fill(new String),this.strings=r):this._$AH=p}_$AI(t,e=this,r,s){let n=this.strings,o=!1;if(n===void 0)t=at(this,t,e,0),o=!Mt(t)||t!==this._$AH&&t!==ot,o&&(this._$AH=t);else{let a=t,i,l;for(t=n[0],i=0;i<n.length-1;i++)l=at(this,a[r+i],e,i),l===ot&&(l=this._$AH[i]),o||(o=!Mt(l)||l!==this._$AH[i]),l===p?t=p:t!==p&&(t+=(l??"")+n[i+1]),this._$AH[i]=l}o&&!s&&this.j(t)}j(t){t===p?this.element.removeAttribute(this.name):this.element.setAttribute(this.name,t??"")}},ds=class extends Lt{constructor(){super(...arguments),this.type=3}j(t){this.element[this.name]=t===p?void 0:t}},ls=class extends Lt{constructor(){super(...arguments),this.type=4}j(t){this.element.toggleAttribute(this.name,!!t&&t!==p)}},cs=class extends Lt{constructor(t,e,r,s,n){super(t,e,r,s,n),this.type=5}_$AI(t,e=this){if((t=at(this,t,e,0)??p)===ot)return;let r=this._$AH,s=t===p&&r!==p||t.capture!==r.capture||t.once!==r.once||t.passive!==r.passive,n=t!==p&&(r===p||s);s&&this.element.removeEventListener(this.name,this,r),n&&this.element.addEventListener(this.name,this,t),this._$AH=t}handleEvent(t){typeof this._$AH=="function"?this._$AH.call(this.options?.host??this.element,t):this._$AH.handleEvent(t)}},us=class{constructor(t,e,r){this.element=t,this.type=6,this._$AN=void 0,this._$AM=e,this.options=r}get _$AU(){return this._$AM._$AU}_$AI(t){at(this,t)}},Pn={M:Ie,P:K,A:Ne,C:1,L:ns,R:as,D:ts,V:at,I:ee,H:Lt,N:ls,U:cs,B:ds,F:us},Wn=Et.litHtmlPolyfillSupport;Wn?.(fe,ee),(Et.litHtmlVersions??(Et.litHtmlVersions=[])).push("3.3.3");var Ce=(t,e,r)=>{let s=r?.renderBefore??e,n=s._$litPart$;if(n===void 0){let o=r?.renderBefore??null;s._$litPart$=n=new ee(e.insertBefore(Nt(),o),o,void 0,r??{})}return n._$AI(t),n};var Tt=globalThis,ft=class extends ht{constructor(){super(...arguments),this.renderOptions={host:this},this._$Do=void 0}createRenderRoot(){var e;let t=super.createRenderRoot();return(e=this.renderOptions).renderBefore??(e.renderBefore=t.firstChild),t}update(t){let e=this.render();this.hasUpdated||(this.renderOptions.isConnected=this.isConnected),super.update(t),this._$Do=Ce(e,this.renderRoot,this.renderOptions)}connectedCallback(){super.connectedCallback(),this._$Do?.setConnected(!0)}disconnectedCallback(){super.disconnectedCallback(),this._$Do?.setConnected(!1)}render(){return ot}};ft._$litElement$=!0,ft.finalized=!0,Tt.litElementHydrateSupport?.({LitElement:ft});var Un=Tt.litElementPolyfillSupport;Un?.({LitElement:ft});(Tt.litElementVersions??(Tt.litElementVersions=[])).push("4.2.2");var bs=["health","system-presence","usage.status","usage.cost","agents.list","sessions.list","sessions.resolve","sessions.get","sessions.usage","sessions.usage.timeseries","sessions.usage.logs","node.list","node.describe","cron.get","cron.list","cron.status","cron.runs","dashboard.connector.list"],hs=["presence","sessions.changed","boardstate.changed"],zn=["sum","avg","min","max","last","count","pick","format"],cr=class extends Error{constructor(e,r){super(r);ct(this,"code");this.code=e,this.name="DashboardBindingResolutionError"}};function jn(t){for(let e of t){let r=e.charCodeAt(0);if(r<32||r===127)return!0}return!1}function Fn(t){if(t.startsWith("/")||/^([a-zA-Z]:[\\/]|[\\/])/.test(t)||jn(t))throw new cr("binding_invalid","file binding path is invalid");let e=t.replaceAll("\\","/").split("/").filter(Boolean);if(e.length===0||e.some(r=>r==="."||r===".."||r.includes(":")))throw new cr("binding_invalid","file binding path is invalid");return e.join("/")}var ps=/^[a-z0-9-]{1,40}$/,Hn=/^(user|system|agent:[A-Za-z0-9._-]{1,64})$/,Vn=/^agent:[A-Za-z0-9._-]{1,64}$/,qn=new Set(["shared","private"]),Gn=/^[A-Za-z0-9:._-]{1,128}$/,Kn=/^[A-Za-z0-9_-]{1,48}$/,Jn=/^builtin:(stat-card|markdown|table|iframe-embed|sessions|usage|cron|instances|activity|chart|notes|action-form|action-button|preview|agent-status|approvals|chat)$/,Xn=/^custom:[A-Za-z0-9._-]{1,64}$/,Yn=/^[A-Za-z0-9._-]{1,64}$/,re=/^[A-Za-z0-9._-]{1,64}$/,Oe=/^[A-Za-z0-9._-]{1,64}$/,ur=/^[A-Za-z0-9._-]{1,64}:[A-Za-z0-9._-]{1,64}$/,br=64,Zn=/^[A-Za-z0-9._+/=-]{1,128}$/,Qn=8*1024,gs=/^[A-Za-z0-9._-]{1,64}$/,to=8*1024,hr=32,fs=/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/,pr=/^[A-Za-z0-9_]{1,32}$/,eo=/\{([A-Za-z0-9_]+)\}/g,gr=2e3,Ut=8,fr=20,ro=1e3,so=["text","number","select"];function Be(t){return typeof t=="object"&&t!==null&&!Array.isArray(t)}function C(t,e){if(!Be(t))throw new Error(`${e} must be an object`);return t}function N(t,e,r){for(let s of Object.keys(t))if(!e.includes(s))throw new Error(`${r}.${s} is not allowed`)}function T(t,e,r){let s=t[e];if(typeof s!="string")throw new Error(`${r}.${e} must be a string`);return s}function P(t,e,r){let s=t[e];if(s!==void 0){if(typeof s!="string")throw new Error(`${r}.${e} must be a string`);return s}}function me(t,e,r){let s=t[e];if(typeof s!="boolean")throw new Error(`${r}.${e} must be a boolean`);return s}function V(t,e){if(!Array.isArray(t))throw new Error(`${e} must be an array`);return t}function Ht(t,e){if(typeof t!="string"||!Hn.test(t))throw new Error(`${e} createdBy is invalid`);return t}function pt(t,e,r,s){if(!Number.isInteger(t)||t<r||t>s)throw new Error(`${e} must be an integer from ${r} to ${s}`);return t}function no(t,e){let r=C(t,e);N(r,["x","y","w","h"],e);let s={x:pt(r.x,`${e}.x`,0,11),y:pt(r.y,`${e}.y`,0,499),w:pt(r.w,`${e}.w`,1,12),h:pt(r.h,`${e}.h`,1,20)};if(s.x+s.w>12)throw new Error(`${e}.x + w must be 12 or less`);return s}function Ct(t,e){if(t===null||typeof t=="string"||typeof t=="boolean"||typeof t=="number"&&Number.isFinite(t))return t;if(Array.isArray(t))return t.map((r,s)=>Ct(r,`${e}[${s}]`));if(Be(t)){let r={};for(let[s,n]of Object.entries(t))r[s]=Ct(n,`${e}.${s}`);return r}throw new Error(`${e} must be JSON-serializable`)}function ms(t){return new TextEncoder().encode(JSON.stringify(t)).length}function oo(t,e){let r=C(t,e),s=T(r,"source",e);if(s==="rpc"){N(r,["source","method"],e);let n=T(r,"method",e);if(!bs.includes(n))throw new Error(`${e}.method is not allowlisted`);return{source:s,method:n}}if(s==="file"){N(r,["source","path","pointer"],e);let n=T(r,"path",e);Fn(n);let o=P(r,"pointer",e);return{source:s,path:n,...o!==void 0?{pointer:o}:{}}}if(s==="static"){N(r,["source","value"],e);let n=Ct(r.value,`${e}.value`);if(ms(n)>to)throw new Error(`${e}.value must serialize to 8 KB or less`);return{source:s,value:n}}if(s==="stream"){N(r,["source","event","pointer"],e);let n=T(r,"event",e);if(!hs.includes(n))throw new Error(`${e}.event is not allowlisted`);let o=P(r,"pointer",e);if(o!==void 0&&!o.startsWith("/"))throw new Error(`${e}.pointer must be a JSON pointer`);return{source:s,event:n,...o!==void 0?{pointer:o}:{}}}if(s==="computed"){N(r,["source","op","inputs","arg"],e);let n=T(r,"op",e);if(!zn.includes(n))throw new Error(`${e}.op is not a valid computed op`);let o=V(r.inputs,`${e}.inputs`);if(o.length<1||o.length>hr)throw new Error(`${e}.inputs must contain 1 to ${hr} entries`);let a=o.map((u,b)=>{if(typeof u!="string"||!gs.test(u))throw new Error(`${e}.inputs[${b}] is invalid`);return u}),i=n==="pick"||n==="format",l=P(r,"arg",e);if(i&&(l===void 0||l.length===0))throw new Error(`${e}.arg is required for the ${n} op`);if(!i&&l!==void 0)throw new Error(`${e}.arg is not allowed for the ${n} op`);if(n==="pick"&&l!==void 0&&!l.startsWith("/"))throw new Error(`${e}.arg must be a JSON pointer for the pick op`);return{source:s,op:n,inputs:a,...l!==void 0?{arg:l}:{}}}if(s==="mcp"){N(r,["source","connector","tool","args"],e);let n=T(r,"connector",e);if(!re.test(n))throw new Error(`${e}.connector is invalid`);let o=T(r,"tool",e);if(!Oe.test(o))throw new Error(`${e}.tool is invalid`);let a=ys(r.args,`${e}.args`);return{source:s,connector:n,tool:o,...a!==void 0?{args:a}:{}}}throw new Error(`${e}.source is invalid`)}function ys(t,e){if(t===void 0)return;let r=Ct(t,e);if(!Be(r))throw new Error(`${e} must be an object`);if(ms(r)>Qn)throw new Error(`${e} must serialize to 8 KB or less`);return r}function ao(t,e){let r=C(t,e),s={};for(let[n,o]of Object.entries(r)){if(!gs.test(n))throw new Error(`${e}.${n} binding id is invalid`);s[n]=oo(o,`${e}.${n}`)}for(let[n,o]of Object.entries(s))if(o.source==="computed")for(let a of o.inputs){let i=s[a];if(!i)throw new Error(`${e}.${n}.inputs references unknown binding: ${a}`);if(i.source==="computed")throw new Error(`${e}.${n}.inputs may not reference another computed binding: ${a}`)}return s}function io(t,e){let r=C(t,e);N(r,["expiresAt"],e);let s=T(r,"expiresAt",e);if(!fs.test(s)||Number.isNaN(Date.parse(s)))throw new Error(`${e}.expiresAt must be an ISO 8601 timestamp`);return{expiresAt:s}}function lo(t,e){let r=C(t,e);N(r,["template","fields","buttonLabel","mode","connector","tool","argsFrom"],e);let s=T(r,"template",e);if(s.length<1||s.length>gr)throw new Error(`${e}.template must be 1-${gr} characters`);let n=V(r.fields,`${e}.fields`);if(n.length<1||n.length>Ut)throw new Error(`${e}.fields must contain 1 to ${Ut} entries`);let o=new Set;if(n.forEach((i,l)=>{let u=`${e}.fields[${l}]`,b=C(i,u);N(b,["name","label","type","options","maxLength"],u);let h=T(b,"name",u);if(!pr.test(h))throw new Error(`${u}.name is invalid`);if(o.has(h))throw new Error(`${u}.name is a duplicate: ${h}`);o.add(h);let m=T(b,"label",u);if(m.length<1||m.length>80)throw new Error(`${u}.label must be 1-80 characters`);let w=T(b,"type",u);if(!so.includes(w))throw new Error(`${u}.type must be text, number, or select`);if(w==="select"){let v=V(b.options,`${u}.options`);if(v.length<1||v.length>fr)throw new Error(`${u}.options must contain 1 to ${fr} entries`);v.forEach((g,_)=>{if(typeof g!="string"||g.length<1||g.length>80)throw new Error(`${u}.options[${_}] must be a 1-80 character string`)})}else if(b.options!==void 0)throw new Error(`${u}.options is only allowed for select fields`);b.maxLength!==void 0&&pt(b.maxLength,`${u}.maxLength`,1,ro)}),r.buttonLabel!==void 0){let i=T(r,"buttonLabel",e);if(i.length<1||i.length>40)throw new Error(`${e}.buttonLabel must be 1-40 characters`)}for(let i of s.matchAll(eo)){let l=i[1];if(!o.has(l))throw new Error(`${e}.template references unknown field: {${l}}`)}let a=P(r,"mode",e);if(a!==void 0&&a!=="prompt"&&a!=="tool")throw new Error(`${e}.mode must be "prompt" or "tool"`);if(a==="tool"){let i=T(r,"connector",e);if(!re.test(i))throw new Error(`${e}.connector is invalid`);let l=T(r,"tool",e);if(!Oe.test(l))throw new Error(`${e}.tool is invalid`);if(r.argsFrom!==void 0){let u=C(r.argsFrom,`${e}.argsFrom`),b=Object.entries(u);if(b.length>Ut)throw new Error(`${e}.argsFrom must contain at most ${Ut} entries`);for(let[h,m]of b){if(!pr.test(h))throw new Error(`${e}.argsFrom key is invalid: ${h}`);if(typeof m!="string"||!o.has(m))throw new Error(`${e}.argsFrom references unknown field: ${String(m)}`)}}}else for(let i of["connector","tool","argsFrom"])if(r[i]!==void 0)throw new Error(`${e}.${i} is only allowed when mode is "tool"`)}function co(t,e){let r=C(t,e);N(r,["connector","tool","args","label"],e);let s=T(r,"connector",e);if(!re.test(s))throw new Error(`${e}.connector is invalid`);let n=T(r,"tool",e);if(!Oe.test(n))throw new Error(`${e}.tool is invalid`);ys(r.args,`${e}.args`);let o=P(r,"label",e);if(o!==void 0&&(o.length<1||o.length>40))throw new Error(`${e}.label must be 1-40 characters`)}function uo(t,e){let r=C(t,e);N(r,["id","kind","title","grid","collapsed","hidden","bindings","props","ephemeral"],e);let s=T(r,"id",e);if(!Kn.test(s))throw new Error(`${e}.id is invalid`);let n=T(r,"kind",e);if(!Jn.test(n)&&!Xn.test(n))throw new Error(`${e}.kind is invalid`);let o=P(r,"title",e);if(o!==void 0&&o.length>80)throw new Error(`${e}.title must be 80 characters or fewer`);let a=r.bindings===void 0?void 0:ao(r.bindings,`${e}.bindings`),i=r.props===void 0?void 0:Ct(r.props,`${e}.props`),l=r.ephemeral===void 0?void 0:io(r.ephemeral,`${e}.ephemeral`);return n==="builtin:action-form"&&lo(i,`${e}.props`),n==="builtin:action-button"&&co(i,`${e}.props`),{id:s,kind:n,...o!==void 0?{title:o}:{},grid:no(r.grid,`${e}.grid`),collapsed:me(r,"collapsed",e),hidden:me(r,"hidden",e),...a!==void 0?{bindings:a}:{},...i!==void 0?{props:i}:{},...l!==void 0?{ephemeral:l}:{}}}function bo(t,e){if(t!==void 0){if(t!=="grid"&&t!=="full")throw new Error(`${e}.layout must be "grid" or "full"`);return t}}function ho(t,e){if(t!==void 0){if(typeof t!="string"||!qn.has(t))throw new Error(`${e}.visibility must be "shared" or "private"`);return t}}function po(t,e){let r=C(t,e);N(r,["slug","title","icon","hidden","layout","createdBy","visibility","owner","widgets"],e);let s=T(r,"slug",e);if(!ps.test(s))throw new Error(`${e}.slug is invalid`);let n=T(r,"title",e);if(n.length<1||n.length>80)throw new Error(`${e}.title must be 1-80 characters`);let o=P(r,"icon",e);if(o!==void 0&&o.length>40)throw new Error(`${e}.icon must be 40 characters or fewer`);let a=bo(r.layout,e),i=ho(r.visibility,e),l=P(r,"owner",e);if(l!==void 0&&!Gn.test(l))throw new Error(`${e}.owner is invalid`);if(i==="private"&&l===void 0)throw new Error(`${e}.owner is required when the tab is private`);let u=V(r.widgets,`${e}.widgets`);if(u.length>24)throw new Error(`${e}.widgets must contain at most 24 entries`);return{slug:s,title:n,...o!==void 0?{icon:o}:{},hidden:me(r,"hidden",e),...a!==void 0?{layout:a}:{},createdBy:Ht(r.createdBy,`${e}.createdBy`),...i==="private"?{visibility:i}:{},...l!==void 0?{owner:l}:{},widgets:u.map((b,h)=>uo(b,`${e}.widgets[${h}]`))}}function go(t,e){let r=C(t,e);N(r,["status","createdBy","approvedBy","approvedAt"],e);let s=T(r,"status",e);if(s!=="pending"&&s!=="approved"&&s!=="rejected")throw new Error(`${e}.status is invalid`);let n=r.approvedBy===void 0?void 0:Ht(r.approvedBy,`${e}.approvedBy`),o=P(r,"approvedAt",e);return{status:s,createdBy:Ht(r.createdBy,`${e}.createdBy`),...n!==void 0?{approvedBy:n}:{},...o!==void 0?{approvedAt:o}:{}}}function fo(t){let e=C(t,"widgetsRegistry"),r={};for(let[s,n]of Object.entries(e)){if(!Yn.test(s))throw new Error(`widgetsRegistry.${s} name is invalid`);r[s]=go(n,`widgetsRegistry.${s}`)}return r}var mo=new Set(["requested","granted","revoked"]);function yo(t,e){let r=C(t,e);N(r,["status","methods","streams","tools","toolsHash","autoConfirm","expiresAt","agents","description","grantedBy","grantedAt"],e);let s=r.status;if(typeof s!="string"||!mo.has(s))throw new Error(`${e}.status must be requested, granted, or revoked`);let n=mr(r.methods,`${e}.methods`,bs,"allowlisted read method"),o=mr(r.streams,`${e}.streams`,hs,"allowlisted stream channel"),a=r.tools===void 0?void 0:V(r.tools,`${e}.tools`).map((v,g)=>{if(typeof v!="string"||v.length>br||!ur.test(v))throw new Error(`${e}.tools[${g}] is not a valid connector:tool id`);return v});if(a!==void 0&&new Set(a).size!==a.length)throw new Error(`${e}.tools contains duplicate tool ids`);let i=P(r,"toolsHash",e);if(i!==void 0&&!Zn.test(i))throw new Error(`${e}.toolsHash is invalid`);let l=r.autoConfirm===void 0?void 0:V(r.autoConfirm,`${e}.autoConfirm`).map((v,g)=>{if(typeof v!="string"||v.length>br||!ur.test(v))throw new Error(`${e}.autoConfirm[${g}] is not a valid connector:tool id`);return v});if(l!==void 0){if(new Set(l).size!==l.length)throw new Error(`${e}.autoConfirm contains duplicate tool ids`);let v=new Set(a??[]);for(let g of l)if(!v.has(g))throw new Error(`${e}.autoConfirm[${g}] is not one of the grant's tools`)}let u=P(r,"expiresAt",e);if(u!==void 0&&(!fs.test(u)||Number.isNaN(Date.parse(u))))throw new Error(`${e}.expiresAt must be an ISO 8601 timestamp`);let b=r.agents===void 0?void 0:V(r.agents,`${e}.agents`).map((v,g)=>{if(typeof v!="string"||!Vn.test(v))throw new Error(`${e}.agents[${g}] is not a valid agent actor`);return v});if(b!==void 0){if(b.length===0)throw new Error(`${e}.agents must be a non-empty array (omit it to allow all agents)`);if(new Set(b).size!==b.length)throw new Error(`${e}.agents contains duplicate actors`)}let h=P(r,"description",e);if(h!==void 0&&h.length>200)throw new Error(`${e}.description must be 200 characters or fewer`);let m=r.grantedBy===void 0?void 0:Ht(r.grantedBy,`${e}.grantedBy`),w=P(r,"grantedAt",e);return{status:s,methods:n,streams:o,...a!==void 0?{tools:a}:{},...i!==void 0?{toolsHash:i}:{},...l!==void 0?{autoConfirm:l}:{},...u!==void 0?{expiresAt:u}:{},...b!==void 0?{agents:b}:{},...h!==void 0?{description:h}:{},...m!==void 0?{grantedBy:m}:{},...w!==void 0?{grantedAt:w}:{}}}function mr(t,e,r,s){return V(t,e).map((n,o)=>{if(typeof n!="string"||!r.includes(n))throw new Error(`${e}[${o}] is not an ${s}`);return n})}function wo(t){if(t===void 0)return{};let e=C(t,"capabilitiesRegistry"),r={};for(let[s,n]of Object.entries(e)){if(!re.test(s))throw new Error(`capabilitiesRegistry.${s} connector name is invalid`);r[s]=yo(n,`capabilitiesRegistry.${s}`)}return r}function vo(t,e){let r=C(t,"prefs");N(r,["tabOrder"],"prefs");let s=V(r.tabOrder,"prefs.tabOrder"),n=new Set;return{tabOrder:s.map((o,a)=>{if(typeof o!="string"||!ps.test(o))throw new Error(`prefs.tabOrder[${a}] is invalid`);if(!e.has(o))throw new Error(`prefs.tabOrder[${a}] is not a tab slug`);if(n.has(o))throw new Error(`prefs.tabOrder contains duplicate slug: ${o}`);return n.add(o),o})}}function _o(t){let e=new Set;for(let r of t){if(e.has(r.slug))throw new Error(`duplicate tab slug: ${r.slug}`);e.add(r.slug)}return e}function xo(t){let e=new Set;for(let r of t)for(let s of r.widgets){if(e.has(s.id))throw new Error(`duplicate widget id: ${s.id}`);e.add(s.id)}}function $o(t){let e=C(t,"workspace");if(N(e,["schemaVersion","workspaceVersion","tabs","widgetsRegistry","capabilitiesRegistry","prefs"],"workspace"),e.schemaVersion!==1)throw new Error("schemaVersion must be 1");let r=pt(e.workspaceVersion,"workspaceVersion",0,Number.MAX_SAFE_INTEGER),s=V(e.tabs,"tabs");if(s.length>32)throw new Error("tabs must contain at most 32 entries");let n=s.map((a,i)=>po(a,`tabs[${i}]`)),o=_o(n);return xo(n),{schemaVersion:1,workspaceVersion:r,tabs:n,widgetsRegistry:fo(e.widgetsRegistry),capabilitiesRegistry:wo(e.capabilitiesRegistry),prefs:vo(e.prefs,o)}}var Ao=/^[A-Za-z0-9._-]{1,64}$/,ko=/^[A-Za-z0-9._-]{1,64}$/,Eo=/^[A-Za-z0-9._-]{1,64}:[A-Za-z0-9._-]{1,64}$/,To=64,yr=80,wr=280,Vt=80,vr=200,_r=16,xr=32;function So(t){return typeof t=="object"&&t!==null&&!Array.isArray(t)}function se(t,e){if(!So(t))throw new Error(`${e} must be an object`);return t}function De(t,e,r){for(let s of Object.keys(t))if(!e.includes(s))throw new Error(`${r}.${s} is not allowed`)}function mt(t,e,r){let s=t[e];if(typeof s!="string")throw new Error(`${r}.${e} must be a string`);return s}function Ro(t,e,r){let s=t[e];if(s!==void 0){if(typeof s!="string")throw new Error(`${r}.${e} must be a string`);return s}}function $r(t,e,r){let s=t[e];if(s!==void 0){if(!Array.isArray(s))throw new Error(`${r}.${e} must be an array`);return s.map((n,o)=>{if(typeof n!="string"||n.length===0)throw new Error(`${r}.${e}[${o}] must be a non-empty string`);return n})}}function Io(t,e,r){let s=se(t,r);De(s,["id","label","readOnly"],r);let n=mt(s,"id",r);if(n.length>To||!Eo.test(n))throw new Error(`${r}.id is not a valid connector:tool id`);if(n.slice(0,n.indexOf(":"))!==e)throw new Error(`${r}.id "${n}" must be namespaced under connector "${e}"`);let o=mt(s,"label",r);if(o.length<1||o.length>Vt)throw new Error(`${r}.label must be 1-${Vt} characters`);let a=s.readOnly;if(a!==void 0&&typeof a!="boolean")throw new Error(`${r}.readOnly must be a boolean`);return{id:n,label:o,...a!==void 0?{readOnly:a}:{}}}function No(t,e,r){let s=se(t,r);De(s,["label","reason","methods","streams","tools"],r);let n=mt(s,"label",r);if(n.length<1||n.length>Vt)throw new Error(`${r}.label must be 1-${Vt} characters`);let o=Ro(s,"reason",r);if(o!==void 0&&o.length>vr)throw new Error(`${r}.reason must be ${vr} characters or fewer`);let a=$r(s,"methods",r),i=$r(s,"streams",r),l;if(s.tools!==void 0){if(!Array.isArray(s.tools))throw new Error(`${r}.tools must be an array`);if(s.tools.length>xr)throw new Error(`${r}.tools must contain at most ${xr} entries`);l=s.tools.map((b,h)=>Io(b,e,`${r}.tools[${h}]`));let u=l.map(b=>b.id);if(new Set(u).size!==u.length)throw new Error(`${r}.tools contains duplicate tool ids`)}if(!((a?.length??0)>0||(i?.length??0)>0||(l?.length??0)>0))throw new Error(`${r} must request at least one tool, method, or stream`);return{label:n,...o!==void 0?{reason:o}:{},...a!==void 0?{methods:a}:{},...i!==void 0?{streams:i}:{},...l!==void 0?{tools:l}:{}}}function Mo(t,e){if(t===void 0)return{};let r=se(t,e);if(Object.keys(r).length>_r)throw new Error(`${e} must reference at most ${_r} connectors`);let s={};for(let[n,o]of Object.entries(r)){if(!ko.test(n))throw new Error(`${e}.${n} connector name is invalid`);s[n]=No(o,n,`${e}.${n}`)}return s}function Co(t){let e=se(t,"recipe");if(De(e,["recipeVersion","name","title","description","doc","grantsManifest"],"recipe"),e.recipeVersion!==1)throw new Error("recipe.recipeVersion must be 1");let r=mt(e,"name","recipe");if(!Ao.test(r))throw new Error("recipe.name is invalid");let s=mt(e,"title","recipe");if(s.length<1||s.length>yr)throw new Error(`recipe.title must be 1-${yr} characters`);let n=mt(e,"description","recipe");if(n.length<1||n.length>wr)throw new Error(`recipe.description must be 1-${wr} characters`);if(e.doc===void 0)throw new Error("recipe.doc is required");return{recipeVersion:1,name:r,title:s,description:n,doc:$o(e.doc),grantsManifest:Mo(e.grantsManifest,"recipe.grantsManifest")}}var Oo="boardstate.chat.event";function z(t,e,r,s){return{x:t,y:e,w:r,h:s}}var Bo=[{kind:"builtin:stat-card",summary:"One number that matters \u2014 a KPI with a label.",bindings:[{key:"value",shape:"number | string, or a structured payload + props.metric"}],props:{format:'"usd" | "int" | "percent" | "raw" (how the number renders)',metric:"when the binding resolves an object, the field name to display",label:"inner label (omit if it would just repeat the title)"},example:{id:"mrr",kind:"builtin:stat-card",title:"MRR",grid:z(0,0,3,2),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:128400}},props:{format:"usd",label:"Monthly recurring revenue"}}},{kind:"builtin:chart",summary:"Trends, comparisons, budgets \u2014 a small inline chart.",bindings:[{key:"value",shape:"number[] (or labeled points {label,value}[])"}],props:{type:'"line" | "bar" | "area" | "sparkline" | "gauge" (default line)',detail:"true adds labeled axes, gridlines, and value tooltips (line/bar/area)",label:"sparkline only: true shows the trailing value as an end label"},example:{id:"revenue-trend",kind:"builtin:chart",title:"Revenue (14d)",grid:z(0,2,8,5),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:[8,12,10,18,24,21,30,35,41,52]}},props:{type:"area"}},examples:[{id:"signups-spark",kind:"builtin:chart",title:"Signups",grid:z(0,7,3,2),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:[12,9,14,11,17,15,22]}},props:{type:"sparkline",label:!0}},{id:"latency-detail",kind:"builtin:chart",title:"p95 latency (ms)",grid:z(0,9,8,5),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:[180,220,190,240,210,260,230]}},props:{type:"line",detail:!0}}]},{kind:"builtin:table",summary:"Rows and columns \u2014 a compact table (keep ~10 visible rows).",bindings:[{key:"rows",shape:"Array<Record<string, unknown>> \u2014 NOT `value`"}],props:{columns:"string[] of keys to show (defaults to the first row's keys)",limit:"max visible rows before a \u201C+N more\u201D count"},example:{id:"recent-runs",kind:"builtin:table",title:"Recent runs",grid:z(0,7,8,4),collapsed:!1,hidden:!1,bindings:{rows:{source:"static",value:[{agent:"finance",task:"Q3 rollup",status:"done"},{agent:"ops",task:"Log sweep",status:"running"}]}},props:{columns:["agent","task","status"]}}},{kind:"builtin:markdown",summary:"Prose, explanations, small markdown tables (sanitized).",bindings:[{key:"content",shape:"markdown string \u2014 NOT `value`"}],props:{markdown:"inline markdown source (used when there is no `content` binding)",text:"alias for `markdown`"},example:{id:"summary",kind:"builtin:markdown",title:"Summary",grid:z(8,2,4,5),collapsed:!1,hidden:!1,props:{markdown:`## Insights

- Signal up **6.5\xD7** across 14 days.
- Momentum late.`}}},{kind:"builtin:notes",summary:"Operator scratch text (persisted via widget state).",bindings:[],props:{text:"starter content"},example:{id:"scratchpad",kind:"builtin:notes",title:"Notes",grid:z(8,7,4,4),collapsed:!1,hidden:!1,props:{text:"Jot findings here\u2026"}}},{kind:"builtin:activity",summary:"An event feed \u2014 recent things that happened.",bindings:[{key:"value",shape:"{ entries: [{ ts, jobName, status, summary }] }"}],props:{limit:"max entries shown"},example:{id:"agent-events",kind:"builtin:activity",title:"Agent events",grid:z(0,11,6,4),collapsed:!1,hidden:!1,bindings:{value:{source:"static",value:{entries:[{ts:17836e8,jobName:"finance",status:"ok",summary:"Rollup posted"}]}}}}},{kind:"builtin:action-form",summary:"The chat\u2194dashboard loop \u2014 a form that submits through the control plane.",bindings:[],props:{template:"the message sent on submit; `{{fieldName}}` interpolates a field (single pass)",fields:'array of { name, label, type: "text"|"number"|"select", options?, maxLength? }',buttonLabel:"the submit button text (optional)",mode:'"prompt" (default: submit the template to the agent) or "tool" (invoke a granted external tool)',connector:"tool mode only: the granted connector name (SPEC \xA717 v2)",tool:"tool mode only: the tool to invoke on that connector",argsFrom:"tool mode only: map of tool-arg name \u2192 declared field name"},example:{id:"ask-agent",kind:"builtin:action-form",title:"Ask the agent",grid:z(0,0,4,3),collapsed:!1,hidden:!1,props:{template:"Summarize {{topic}} for the board.",fields:[{name:"topic",label:"Topic",type:"text"}],buttonLabel:"Ask"}},examples:[{id:"file-ticket",kind:"builtin:action-form",title:"File a ticket",grid:z(0,0,4,4),collapsed:!1,hidden:!1,props:{mode:"tool",connector:"linear",tool:"create_issue",template:"Create issue: {title}",fields:[{name:"title",label:"Title",type:"text",maxLength:120},{name:"priority",label:"Priority",type:"select",options:["low","med","high"]}],argsFrom:{title:"title",priority:"priority"},buttonLabel:"Create"}}]},{kind:"builtin:action-button",summary:"One click \u2192 invoke a granted external tool with fixed args (operator-confirmed).",bindings:[],props:{connector:"the granted connector name (SPEC \xA717 v2)",tool:"the tool to invoke on that connector",args:"fixed argument object passed on click (optional)",label:"button text (optional)"},example:{id:"restart-worker",kind:"builtin:action-button",title:"Restart worker",grid:z(0,0,3,2),collapsed:!1,hidden:!1,props:{connector:"officecli",tool:"restart_service",args:{service:"worker"},label:"Restart"}}},{kind:"builtin:chat",summary:"Talk to the agent and watch it work (ignores bindings).",bindings:[],props:{placeholder:"empty-input hint text"},example:{id:"assistant",kind:"builtin:chat",title:"Assistant",grid:z(0,0,6,8),collapsed:!1,hidden:!1,props:{placeholder:"Ask me to build a view\u2026"}}}],Do=[{kind:"builtin:sessions",summary:"Who/what is running.",valueShape:"rows { key, label, status, hasActiveRun, updatedAt }; props.limit"},{kind:"builtin:agent-status",summary:"Agents + goals/progress.",valueShape:"sessions shape + goal { objective, tokensUsed, tokenBudget }"},{kind:"builtin:usage",summary:"Cost/token totals.",valueShape:"{ totals: { totalCost, totalTokens }, days? }"},{kind:"builtin:cron",summary:"Scheduled jobs.",valueShape:"{ jobs: [{ id, name, enabled, state: { nextRunAtMs, lastRunStatus } }] }"},{kind:"builtin:instances",summary:"Fleet presence.",valueShape:"{ presence: [{ instanceId, platform, version, lastInputSeconds }] }"},{kind:"builtin:approvals",summary:"Pending widget approvals (reads the live registry; ignores bindings).",valueShape:"none \u2014 reads the registry"},{kind:"builtin:preview",summary:"A live page preview.",valueShape:"props.url (same-origin ok; cross-origin needs host opt-in)"},{kind:"builtin:iframe-embed",summary:"An embedded live page.",valueShape:"props.url (same-origin ok; cross-origin needs host opt-in)"}];[...Bo.map(t=>t.kind),...Do.map(t=>t.kind)];function Ar(t){let e=new Map;for(let r of t.tabs)for(let s of r.widgets)e.set(s.id,{widget:s,tabSlug:r.slug});return e}function kr(t){return new Map(t.tabs.map(e=>[e.slug,e]))}function Lo(t,e){return t.grid.x===e.grid.x&&t.grid.y===e.grid.y&&t.grid.w===e.grid.w&&t.grid.h===e.grid.h}function Po(t,e){let r=[],s=kr(t),n=kr(e);for(let[i,l]of n)s.has(i)||r.push({kind:"tab-added",actor:l.createdBy??null,id:i,label:l.title});for(let[i,l]of s)if(!n.has(i))r.push({kind:"tab-removed",actor:l.createdBy??null,id:i,label:l.title});else{let u=n.get(i);u.title!==l.title&&r.push({kind:"tab-retitled",actor:u.createdBy??l.createdBy??null,id:i,label:u.title,detail:`${l.title} \u2192 ${u.title}`})}let o=Ar(t),a=Ar(e);for(let[i,l]of a)o.has(i)||r.push({kind:"widget-added",actor:l.widget.createdBy??null,id:i,label:l.widget.title||i});for(let[i,l]of o){let u=a.get(i);if(!u){r.push({kind:"widget-removed",actor:l.widget.createdBy??null,id:i,label:l.widget.title||i});continue}let b=l.widget,h=u.widget;(l.tabSlug!==u.tabSlug||!Lo(b,h))&&r.push({kind:"widget-moved",actor:h.createdBy??null,id:i,label:h.title||i,detail:l.tabSlug!==u.tabSlug?`${l.tabSlug} \u2192 ${u.tabSlug}`:void 0}),b.title!==h.title&&r.push({kind:"widget-retitled",actor:h.createdBy??null,id:i,label:h.title||i,detail:`${b.title||i} \u2192 ${h.title||i}`})}return r}function Wo(t){let e=new Map;for(let r of t){let s=e.get(r.actor);s?s.push(r):e.set(r.actor,[r])}return[...e.entries()].map(([r,s])=>({actor:r,entries:s}))}function Uo(t,e){return t.tabs.some(r=>r.widgets.some(s=>s.id===e))}function zo(t,e){let r=e.filter(n=>Uo(n.workspace,t)).map(n=>n.version).toSorted((n,o)=>n-o);if(r.length===0)return;let s=r[0];return e.some(n=>n.version<s)?s:void 0}function X(t){if(typeof t!="string")return null;let e=t.trim();return e.startsWith("agent:")?e.slice(6)||"agent":null}function O(t){return!!t&&typeof t=="object"&&!Array.isArray(t)}function St(t,e=""){return typeof t=="string"?t:e}function gt(t,e=0){return typeof t=="number"&&Number.isFinite(t)?t:e}function jo(t){let e=O(t)?t:{},r=Math.min(12,Math.max(1,Math.trunc(gt(e.w,4)))),s=Math.max(1,Math.trunc(gt(e.h,2)));return{x:Math.min(12-r,Math.max(0,Math.trunc(gt(e.x,0)))),y:Math.max(0,Math.trunc(gt(e.y,0))),w:r,h:s}}function Fo(t){if(!O(t))return null;let e=t.source;return e!=="rpc"&&e!=="file"&&e!=="static"&&e!=="stream"&&e!=="computed"&&e!=="mcp"?null:{source:e,...typeof t.method=="string"?{method:t.method}:{},...typeof t.path=="string"?{path:t.path}:{},...typeof t.pointer=="string"?{pointer:t.pointer}:{},...O(t.params)?{params:t.params}:{},..."value"in t?{value:t.value}:{},...typeof t.event=="string"?{event:t.event}:{},...typeof t.op=="string"?{op:t.op}:{},...Array.isArray(t.inputs)?{inputs:t.inputs.filter(r=>typeof r=="string")}:{},...typeof t.arg=="string"?{arg:t.arg}:{},...typeof t.connector=="string"?{connector:t.connector}:{},...typeof t.tool=="string"?{tool:t.tool}:{},...O(t.args)?{args:t.args}:{}}}function Er(t){if(!O(t))return;let e={};for(let[r,s]of Object.entries(t)){let n=Fo(s);n&&(e[r]=n)}return Object.keys(e).length?e:void 0}function Ho(t){if(!O(t))return null;let e=St(t.id).trim(),r=St(t.kind).trim();if(!e||!r)return null;let s=Vo(t.ephemeral);return{id:e,kind:r,title:St(t.title),grid:jo(t.grid),collapsed:t.collapsed===!0,...typeof t.createdBy=="string"?{createdBy:t.createdBy}:{},...Er(t.bindings)?{bindings:Er(t.bindings)}:{},...O(t.props)?{props:t.props}:{},...s?{ephemeral:s}:{}}}function Vo(t){return!O(t)||typeof t.expiresAt!="string"||!t.expiresAt.trim()?null:{expiresAt:t.expiresAt}}function qo(t){if(!O(t))return null;let e=St(t.slug).trim();if(!e)return null;let r=Array.isArray(t.widgets)?t.widgets.map(Ho).filter(s=>s!==null):[];return{slug:e,title:St(t.title,e),hidden:t.hidden===!0,widgets:r,...t.layout==="full"||t.layout==="grid"?{layout:t.layout}:{},...t.visibility==="private"?{visibility:"private"}:{},...typeof t.owner=="string"?{owner:t.owner}:{},...typeof t.icon=="string"?{icon:t.icon}:{},...typeof t.createdBy=="string"?{createdBy:t.createdBy}:{}}}var Go=new Set(["pending","approved","rejected"]);function Ko(t){if(!O(t))return null;let e=t.status;return typeof e!="string"||!Go.has(e)?null:{status:e,...typeof t.createdBy=="string"?{createdBy:t.createdBy}:{},...typeof t.approvedBy=="string"?{approvedBy:t.approvedBy}:{},...typeof t.approvedAt=="string"?{approvedAt:t.approvedAt}:{}}}function Jo(t){if(!O(t))return{};let e={};for(let[r,s]of Object.entries(t)){let n=Ko(s);n&&(e[r]=n)}return e}var Xo=new Set(["requested","granted","revoked"]);function Yo(t){if(!O(t))return null;let e=t.status;if(typeof e!="string"||!Xo.has(e))return null;let r=s=>Array.isArray(s)?s.filter(n=>typeof n=="string"):[];return{status:e,methods:r(t.methods),streams:r(t.streams),...Array.isArray(t.tools)?{tools:r(t.tools)}:{},...typeof t.toolsHash=="string"?{toolsHash:t.toolsHash}:{},...Array.isArray(t.autoConfirm)?{autoConfirm:r(t.autoConfirm)}:{},...typeof t.expiresAt=="string"?{expiresAt:t.expiresAt}:{},...Array.isArray(t.agents)?{agents:r(t.agents)}:{},...typeof t.description=="string"?{description:t.description}:{},...typeof t.grantedBy=="string"?{grantedBy:t.grantedBy}:{},...typeof t.grantedAt=="string"?{grantedAt:t.grantedAt}:{}}}function Zo(t){if(!O(t))return{};let e={};for(let[r,s]of Object.entries(t)){let n=Yo(s);n&&(e[r]=n)}return e}function ws(t){let e=O(t)?t:{},r=Array.isArray(e.tabs)?e.tabs.map(qo).filter(o=>o!==null):[],s=O(e.prefs)?e.prefs:{},n=Array.isArray(s.tabOrder)?s.tabOrder.filter(o=>typeof o=="string"):[];return{schemaVersion:gt(e.schemaVersion,1),workspaceVersion:gt(e.workspaceVersion,0),tabs:r,prefs:{tabOrder:n},widgetsRegistry:Jo(e.widgetsRegistry),capabilitiesRegistry:Zo(e.capabilitiesRegistry)}}function Le(t){return t.startsWith("custom:")&&t.slice(7)||null}function vs(t,e){let r=Le(e);return r?t.widgetsRegistry[r]?.status??null:null}function ne(t){let e=new Map(t.tabs.map(n=>[n.slug,n])),r=[],s=new Set;for(let n of t.prefs.tabOrder){let o=e.get(n);o&&!s.has(n)&&(r.push(o),s.add(n))}for(let n of t.tabs)s.has(n.slug)||(r.push(n),s.add(n.slug));return r}function oe(t){return ne(t).filter(e=>!e.hidden)}function Qo(t){return ne(t).filter(e=>e.hidden)}function ta(t){let e=[],r=new Map;for(let s of t){let n=X(s.createdBy),o=n?"agent":s.createdBy==="system"?"system":"user",a=o==="agent"?`agent:${n}`:o,i=r.get(a);i||(i={key:a,kind:o,agentId:o==="agent"?n:null,tabs:[]},r.set(a,i),e.push(i)),i.tabs.push(s)}return e}function Pe(t,e){if(e)return t.tabs.find(r=>r.slug===e)}function _s(t,e){let r=Pe(t,e);if(r)return r.slug;let s=oe(t);return s.length>0?s[0].slug:ne(t)[0]?.slug??null}function ae(t,e){if(!e)return t;let r=e.split("/").slice(1).map(n=>n.replace(/~1/g,"/").replace(/~0/g,"~")),s=t;for(let n of r)if(Array.isArray(s)){let o=Number(n);s=Number.isInteger(o)?s[o]:void 0}else if(O(s))s=s[n];else return;return s}var it=new Map,Tr=0;function ea(){return Tr+=1,`sub_${Tr}`}function ra(t){let{tabSlug:e,channel:r,subscriberId:s,deliver:n}=t,o=it.get(e);o||(o=new Map,it.set(e,o));let a=o.get(r);return a||(a=new Map,o.set(r,a)),a.set(s,{subscriberId:s,channel:r,deliver:n}),()=>sa({tabSlug:e,channel:r,subscriberId:s})}function sa(t){let{tabSlug:e,channel:r,subscriberId:s}=t,n=it.get(e),o=n?.get(r);o&&(o.delete(s),o.size===0&&n?.delete(r),n&&n.size===0&&it.delete(e))}function na(t,e){let r=it.get(t);if(r){for(let[s,n]of r)n.delete(e)&&n.size===0&&r.delete(s);r.size===0&&it.delete(t)}}function oa(t){let{tabSlug:e,channel:r,fromSubscriberId:s,payload:n}=t,o=it.get(e)?.get(r);if(!o)return 0;let a=0;for(let i of Array.from(o.values()))i.subscriberId!==s&&(i.deliver(r,n),a+=1);return a}function aa(t){return Math.max(1,(t.width-132)/12)}function ye(t,e,r){return Math.min(r,Math.max(e,t))}function Sr(t,e){return e<=0?0:Math.round(t/(e+12))}function qt(t){let e=ye(t.w,1,12),r=Math.max(1,t.h);return{x:ye(t.x,0,12-e),y:Math.max(0,t.y),w:e,h:r}}function ia(t,e){return t.x<e.x+e.w&&e.x<t.x+t.w&&t.y<e.y+e.h&&e.y<t.y+t.h}function xs(t,e){return t.filter(r=>r.id!==e).map(r=>r.grid)}function We(t,e,r){return xs(e,r).some(s=>ia(t,s))}function da(t){return{widgetId:t.widget.id,mode:t.mode,originRect:{...t.widget.grid},originClientX:t.clientX,originClientY:t.clientY,ghostRect:{...t.widget.grid},pointerDx:0,pointerDy:0,columnWidth:aa(t.metrics)}}function la(t,e,r){t.pointerDx=e-t.originClientX,t.pointerDy=r-t.originClientY;let s=56,n=Sr(e-t.originClientX,t.columnWidth),o=Sr(r-t.originClientY,s),a=qt(t.mode==="move"?{x:t.originRect.x+n,y:t.originRect.y+o,w:t.originRect.w,h:t.originRect.h}:{x:t.originRect.x,y:t.originRect.y,w:t.originRect.w+n,h:t.originRect.h+o});return t.ghostRect=a,a}function Rr(t){let e=qt(t.requested);return We(e,t.widgets,t.widgetId)?ca(e,t.widgets,t.widgetId):e}function ca(t,e,r){let s=ye(t.w,1,12),n=Math.max(1,t.h),o=12-s,a=xs(e,r).reduce((u,b)=>Math.max(u,b.y+b.h),0),i=Math.max(t.y,a)+n,l=null;for(let u=0;u<=i;u+=1){for(let b=0;b<=o;b+=1){let h={x:b,y:u,w:s,h:n};if(We(h,e,r))continue;let m=Math.abs(b-t.x)+Math.abs(u-t.y);(!l||m<l.distance)&&(l={rect:h,distance:m})}if(l&&u>=t.y)break}return l?.rect??null}function Ue(t){return[`grid-column: ${t.x+1} / span ${t.w}`,`grid-row: ${t.y+1} / span ${t.h}`].join("; ")}function $s(t){return t.reduce((e,r)=>Math.max(e,r.grid.y+r.grid.h),0)}function ua(t,e,r){if(e==="move"){let a=r==="left"?-1:r==="right"?1:0,i=r==="up"?-1:r==="down"?1:0;return qt({...t,x:t.x+a,y:t.y+i})}let n=r==="left"?-1:r==="right"?1:0,o=r==="up"?-1:r==="down"?1:0;return qt({...t,w:t.w+n,h:t.h+o})}var As="pending";function W(t){return typeof t=="object"&&t!==null&&!Array.isArray(t)}function ba(t){return W(t)?W(t.doc)?t.doc:W(t.workspace)?t.workspace:t:{}}function ha(t=new Date){return`dashboard-workspace-${t.toISOString().replace(/[:.]/g,"-")}.json`}function pa(t){return typeof t!="string"||!t.startsWith("custom:")?null:t.slice(7)||null}function ks(t){let e=new Set;if(!Array.isArray(t))return e;for(let r of t){let s=W(r)&&Array.isArray(r.widgets)?r.widgets:[];for(let n of s){let o=W(n)?pa(n.kind):null;o&&e.add(o)}}return e}function ga(t,e){if(!W(e))return{};let r=ks(t),s={};for(let[n,o]of Object.entries(e))r.has(n)&&(s[n]=o);return s}function fa(t,e={}){let r=structuredClone(t),s=e.slugs;if(!s||s.length===0)return r;let n=new Set(s),o=Array.isArray(r.tabs)?r.tabs.filter(l=>W(l)&&n.has(l.slug)):[];r.tabs=o;let a=W(r.prefs)?r.prefs:{},i=Array.isArray(a.tabOrder)?a.tabOrder:[];return r.prefs={...a,tabOrder:i.filter(l=>typeof l=="string"&&n.has(l))},r.widgetsRegistry=ga(o,r.widgetsRegistry),r}function ma(t,e={}){return`${JSON.stringify(fa(t,e),null,2)}
`}function ya(t){try{return JSON.parse(t)}catch{throw new Error("Import file is not valid JSON.")}}function wa(t){let e=W(t)&&typeof t.createdBy=="string"?t.createdBy:"user";return{status:As,createdBy:e}}function Es(t){if(!W(t))throw new Error("Import file must be a workspace object.");let e=structuredClone(t),r=W(e.widgetsRegistry)?e.widgetsRegistry:{},s={};for(let[a,i]of Object.entries(r))s[a]=wa(i);for(let a of ks(e.tabs))s[a]??(s[a]={status:As,createdBy:"user"});e.widgetsRegistry=s;let n=W(e.capabilitiesRegistry)?e.capabilitiesRegistry:{},o={};for(let[a,i]of Object.entries(n))if(W(i)){let{grantedBy:l,grantedAt:u,autoConfirm:b,expiresAt:h,agents:m,...w}=i;o[a]={...w,status:"requested"}}return e.capabilitiesRegistry=o,e}function va(t){let e=t.reason?.trim();return e&&e.length>0?e.slice(0,200):void 0}function _a(t){let e=structuredClone(t.doc),r={};for(let[s,n]of Object.entries(t.grantsManifest)){let o=va(n),a=(n.tools??[]).map(i=>i.id);r[s]={status:"requested",methods:n.methods??[],streams:n.streams??[],...a.length>0?{tools:a}:{},...o!==void 0?{description:o}:{}}}return e.capabilitiesRegistry=r,e}function xa(t){return Es(_a(t))}var $a=512*1024,Aa=512*1024,Ts=256*1024,Ss=/^[A-Za-z0-9._-]{1,64}$/;function tt(t){return typeof t=="object"&&t!==null&&!Array.isArray(t)}function ka(t){return new TextEncoder().encode(t).length}function Ea(t,e){let r;try{r=JSON.parse(t)}catch{throw new Error("The gallery index is not valid JSON.")}let s=Array.isArray(r)?r:tt(r)&&Array.isArray(r.widgets)?r.widgets:null;if(!s)throw new Error("The gallery index must be a list of widgets.");let n=[];for(let o of s){if(!tt(o))continue;let a=typeof o.name=="string"?o.name.trim():"",i=typeof o.manifestUrl=="string"?o.manifestUrl.trim():"";if(!Ss.test(a)||!i)continue;let l;try{l=new URL(i,e).toString()}catch{continue}n.push({name:a,description:typeof o.description=="string"?o.description:"",manifestUrl:l})}return n}function Ta(t){return Array.isArray(t)?t.filter(e=>e==="data:read"||e==="prompt:send"):[]}function Sa(t){return Array.isArray(t)?t.map(e=>tt(e)&&typeof e.id=="string"?e.id:null).filter(e=>e!==null):[]}function Ra(t){let e;try{e=JSON.parse(t)}catch{throw new Error("The widget bundle is not valid JSON.")}if(!tt(e)||!tt(e.manifest)||!tt(e.files))throw new Error("The widget bundle must be an object with `manifest` and `files`.");let r=e.manifest,s=typeof r.name=="string"?r.name.trim():"";if(!Ss.test(s))throw new Error("The widget bundle manifest has an invalid name.");let n={};for(let[o,a]of Object.entries(e.files)){if(typeof a!="string")throw new Error("Every widget bundle file must be text.");n[o]=a}return{name:s,title:typeof r.title=="string"?r.title:s,capabilities:Ta(r.capabilities),bindingIds:Sa(r.bindings),manifest:r,files:n}}var Ia=/^[A-Za-z0-9._-]{1,64}$/;function Na(t,e){let r;try{r=JSON.parse(t)}catch{throw new Error("The gallery index is not valid JSON.")}let s=tt(r)&&Array.isArray(r.recipes)?r.recipes:null;if(!s)return[];let n=[];for(let o of s){if(!tt(o))continue;let a=typeof o.name=="string"?o.name.trim():"",i=typeof o.manifestUrl=="string"?o.manifestUrl.trim():"";if(!Ia.test(a)||!i)continue;let l;try{l=new URL(i,e).toString()}catch{continue}let u=Array.isArray(o.connectors)?o.connectors.filter(b=>typeof b=="string"):[];n.push({name:a,title:typeof o.title=="string"&&o.title?o.title:a,description:typeof o.description=="string"?o.description:"",manifestUrl:l,connectors:u})}return n}function Ma(t){let e;try{e=JSON.parse(t)}catch{throw new Error("The recipe bundle is not valid JSON.")}try{return Co(e)}catch(r){throw new Error(`The recipe bundle is invalid: ${r instanceof Error?r.message:String(r)}`)}}function R(t){return t.props??{}}function k(t){return typeof t=="object"&&t!==null&&!Array.isArray(t)}function S(t){if(typeof t=="number")return Number.isFinite(t)?t:void 0;if(typeof t=="string"&&t.trim()){let e=Number(t);return Number.isFinite(e)?e:void 0}}function Ca(t,e){if(!k(t))return;let r=k(t.totals)?t.totals:void 0;switch(e){case"todayCost":return r?.totalCost??t.totalCost;case"todayTokens":return r?.totalTokens??t.totalTokens;default:return t[e]}}function Oa(t,e){if(t==null)return null;let r=S(t);return e==="usd"&&r!==void 0?new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(r):e==="percent"&&r!==void 0?new Intl.NumberFormat("en-US",{style:"percent",maximumFractionDigits:1}).format(r):(e==="int"||e==="integer")&&r!==void 0?new Intl.NumberFormat("en-US",{maximumFractionDigits:0}).format(r):typeof t=="string"?t:r!==void 0?new Intl.NumberFormat("en-US").format(r):JSON.stringify(t)}function Ba(t,e){let r=R(t),s=typeof r.metric=="string"?r.metric:null,n=s?Ca(e,s):e,o=n!==void 0?n:r.value,a=typeof r.label=="string"?r.label:t.title,i=a&&a!==t.title?a:null;return{display:Oa(o,r.format),label:i}}function Da(t,e){let r=R(t);return typeof e=="string"?e:typeof r.markdown=="string"?r.markdown:typeof r.text=="string"?r.text:""}var La=8;function Pa(t,e){return(Array.isArray(e)?e:k(e)&&Array.isArray(e.rows)?e.rows:Array.isArray(R(t).rows)?R(t).rows:[]).filter(k)}function Wa(t,e){let r=R(t).columns;if(Array.isArray(r)){let s=r.filter(n=>typeof n=="string");if(s.length>0)return s}return e.length>0?Object.keys(e[0]):[]}function Ua(t){let e=R(t).limit;return typeof e=="number"&&Number.isFinite(e)&&e>0?Math.min(Math.trunc(e),100):La}function za(t,e){let r=Pa(t,e),s=Ua(t),n=r.slice(0,s);return{columns:Wa(t,n),rows:n,shown:n.length,total:r.length}}var ja=6;function Fa(t){return t.status&&t.status!=="running"?!1:typeof t.hasActiveRun=="boolean"?t.hasActiveRun:t.status==="running"}function Ha(t,e){let r=t.displayName??t.label??t.subject??t.channel;return typeof r=="string"&&r.trim()?r:e}function Va(t,e){let r=Array.isArray(e)?e:k(e)&&Array.isArray(e.sessions)?e.sessions:[],s=S(R(t).limit),n=s&&s>0?Math.trunc(s):ja,o=r.filter(k);return{rows:o.map(a=>{let i=typeof a.key=="string"?a.key:"";return{key:i,label:Ha(a,i),active:Fa({hasActiveRun:typeof a.hasActiveRun=="boolean"?a.hasActiveRun:void 0,status:typeof a.status=="string"?a.status:void 0}),updatedAt:S(a.updatedAt)??null}}).filter(a=>a.key).slice(0,n),total:o.length}}function qa(t,e){let r=k(e)&&k(e.totals)?e.totals:{};return{cost:S(r.totalCost)??0,tokens:S(r.totalTokens)??0,days:k(e)?S(e.days)??null:null}}var Ga=8;function Ka(t){if(!t)return null;let e=t.lastRunStatus??t.lastStatus;return typeof e=="string"?e:null}function Ja(t,e){let r=k(e)&&Array.isArray(e.jobs)?e.jobs:[],s=S(R(t).limit),n=s&&s>0?Math.trunc(s):Ga,o=r.filter(k);return{jobs:o.map(a=>{let i=k(a.state)?a.state:void 0;return{id:typeof a.id=="string"?a.id:"",name:typeof a.name=="string"&&a.name.trim()?a.name:a.id||"",enabled:a.enabled!==!1,nextRunAtMs:i?S(i.nextRunAtMs)??null:null,lastStatus:Ka(i)}}).filter(a=>a.id).slice(0,n),total:o.length}}var Xa=8,Ya=120;function Za(t){let e=t.instanceId??t.host??t.ip??t.deviceFamily;return typeof e=="string"&&e.trim()?e:""}function Qa(t){let e=[t.mode,t.platform,t.version].filter(r=>typeof r=="string"&&r.trim().length>0);return e.length>0?e.join(" \xB7 "):null}function ti(t,e){let r=Array.isArray(e)?e:k(e)&&Array.isArray(e.presence)?e.presence:k(e)&&Array.isArray(e.nodes)?e.nodes:[],s=S(R(t).limit),n=s&&s>0?Math.trunc(s):Xa,o=r.filter(k);return{instances:o.map(a=>{let i=S(a.lastInputSeconds);return{id:Za(a),detail:Qa(a),healthy:i===void 0||i<=Ya,lastInputMs:i!==void 0?i*1e3:null}}).filter(a=>a.id).slice(0,n),total:o.length}}var ei=20;function Ir(t,e=120){return t.length<=e?t:`${t.slice(0,Math.max(0,e-1))}\u2026`}function ri(t){let e=t.jobName??t.jobId??t.action;return typeof e=="string"&&e.trim()?e:"run"}function si(t,e){let r=k(e)&&Array.isArray(e.entries)?e.entries:[],s=S(R(t).limit),n=s&&s>0?Math.trunc(s):ei,o=r.filter(k);return{entries:o.map(a=>({ts:S(a.ts)??null,title:ri(a),detail:typeof a.summary=="string"&&a.summary.trim()?Ir(a.summary,120):typeof a.error=="string"&&a.error.trim()?Ir(a.error,120):null,status:typeof a.status=="string"?a.status:null})).slice(0,n),total:o.length}}function Rs(t,e,r){if(typeof t!="string"||!t.trim())return{status:"missing"};let s=t.trim(),n=globalThis.location?.origin,o=r??n,a;try{a=o?new URL(s,o):new URL(s)}catch{return{status:"ok",url:s,external:!1}}if(a.protocol!=="http:"&&a.protocol!=="https:")return{status:"blocked",reason:"scheme",url:s};let i=o?a.origin!==new URL(o).origin:!0;return i&&!e.allowExternalEmbedUrls?{status:"blocked",reason:"external",url:s}:{status:"ok",url:s,external:i}}var ni=["line","bar","area","sparkline","gauge"],oi="line";function ai(t){if(typeof t=="number")return Number.isFinite(t)?t:void 0;if(k(t))return S(t.y)??S(t.value)}function ii(t){let e=Array.isArray(t)?t:k(t)&&Array.isArray(t.points)?t.points:[],r=[];for(let s of e){let n=ai(s);n!==void 0&&r.push(n)}return r}function di(t){let e=t.type;return typeof e=="string"&&ni.includes(e)?e:oi}function li(t,e){let r=R(t),s=ii(e),n=s.length?Math.min(...s):0,o=s.length?Math.max(...s):0;return{type:di(r),values:s,min:n,max:o,detail:r.detail===!0,label:r.label===!0}}function ci(t){return typeof t=="string"?t:""}var ui=/\{([A-Za-z0-9_]+)\}/g,bi=new Set(["text","number","select"]);function hi(t){if(!k(t))return null;let{name:e,label:r,type:s}=t;if(typeof e!="string"||!e||typeof r!="string"||!r||typeof s!="string"||!bi.has(s))return null;let n=s==="select"&&Array.isArray(t.options)?t.options.filter(a=>typeof a=="string"):void 0;if(s==="select"&&(!n||n.length===0))return null;let o=typeof t.maxLength=="number"&&Number.isInteger(t.maxLength)&&t.maxLength>0?t.maxLength:void 0;return{name:e,label:r,type:s,...n?{options:n}:{},...o!==void 0?{maxLength:o}:{}}}function pi(t){if(!k(t))return{};let e={};for(let[r,s]of Object.entries(t))typeof s=="string"&&(e[r]=s);return e}function gi(t){let e=R(t),r=typeof e.template=="string"?e.template:"",s=Array.isArray(e.fields)?e.fields.map(hi).filter(o=>o!==null):[],n=typeof e.buttonLabel=="string"?e.buttonLabel:null;return(e.mode==="tool"?"tool":"prompt")!="tool"?{template:r,fields:s,buttonLabel:n,mode:"prompt",connector:null,tool:null,argsFrom:null}:{template:r,fields:s,buttonLabel:n,mode:"tool",connector:typeof e.connector=="string"?e.connector:null,tool:typeof e.tool=="string"?e.tool:null,argsFrom:pi(e.argsFrom)}}function Is(t,e){let r=t.maxLength&&t.maxLength>0?t.maxLength:200;if(t.type==="number"){let s=e.trim();return s&&Number.isFinite(Number(s))?s.slice(0,r):""}return t.type==="select"?t.options?.includes(e)?e:"":e.slice(0,r)}function fi(t,e){let r=new Map(t.fields.map(s=>[s.name,s]));return t.template.replace(ui,(s,n)=>{let o=r.get(n);return o?Is(o,e[n]??""):s})}function mi(t,e){let r=new Map(t.fields.map(n=>[n.name,n])),s={};for(let[n,o]of Object.entries(t.argsFrom??{})){let a=r.get(o);a&&(s[n]=Is(a,e[o]??""))}return s}function Nr(t){let e=R(t);return{connector:typeof e.connector=="string"?e.connector:"",tool:typeof e.tool=="string"?e.tool:"",args:k(e.args)?e.args:null,label:typeof e.label=="string"?e.label:null}}var yi=["desktop","tablet","mobile"];function wi(t){let e=R(t).defaultViewport;return typeof e=="string"&&yi.includes(e)?e:"desktop"}var vi=8;function _i(t){return t.status&&t.status!=="running"?!1:typeof t.hasActiveRun=="boolean"?t.hasActiveRun:t.status==="running"}function xi(t,e){return t.length<=e?t:`${t.slice(0,Math.max(0,e-1))}\u2026`}function $i(t,e){let r=t.displayName??t.label??t.subject??t.channel;return typeof r=="string"&&r.trim()?r:e}function Ai(t){let e=k(t.goal)?t.goal:void 0,r=e&&typeof e.objective=="string"?e.objective.trim():"";return r?xi(r,100):null}function ki(t){let e=k(t.goal)?t.goal:void 0;if(!e)return null;let r=S(e.tokensUsed),s=S(e.tokenBudget);return r===void 0||s===void 0||s<=0?null:Math.min(1,Math.max(0,r/s))}function Ei(t,e){let r=Array.isArray(e)?e:k(e)&&Array.isArray(e.sessions)?e.sessions:[],s=S(R(t).limit),n=s&&s>0?Math.trunc(s):vi,o=r.filter(k).map(i=>{let l=typeof i.key=="string"?i.key:"";return{key:l,label:$i(i,l),active:_i({hasActiveRun:typeof i.hasActiveRun=="boolean"?i.hasActiveRun:void 0,status:typeof i.status=="string"?i.status:void 0}),task:Ai(i),progress:ki(i)}}).filter(i=>i.key),a=o.filter(i=>i.active).length;return{rows:o.slice(0,n),activeCount:a,total:o.length}}var Ti=8;function Ns(t){return t==="approve"?"approved":"rejected"}function Si(t,e){return{pending:Object.entries(t.widgetsRegistry).filter(([,r])=>r.status==="pending").map(([r,s])=>({id:r,kind:"widget",title:r,requestedBy:X(s.createdBy)})),onDecide:(r,s)=>e(r.id,Ns(s))}}function Ri(t,e,r,s){let n=Si(t,e).pending,o=Object.entries(t.capabilitiesRegistry??{}),a=u=>{let b=u.tools??[],h=[u.methods.length?`${u.methods.length} read${u.methods.length===1?"":"s"}`:null,u.streams.length?`${u.streams.length} stream${u.streams.length===1?"":"s"}`:null,b.length?`${b.length} tool${b.length===1?"":"s"}`:null].filter(Boolean);return u.description??(h.length?`wants ${h.join(" + ")}`:"data access")},i=o.filter(([,u])=>u.status==="requested").map(([u,b])=>({id:u,kind:"capability",title:u,requestedBy:null,detail:a(b),...(b.tools??[]).length?{tools:b.tools}:{}})),l=o.filter(([,u])=>u.status==="granted"&&((u.tools??[]).length>0||u.expiresAt)).map(([u,b])=>({id:u,kind:"capability",title:u,requestedBy:null,granted:!0,detail:a(b),...(b.tools??[]).length?{tools:b.tools}:{},...(b.autoConfirm??[]).length?{autoConfirm:b.autoConfirm}:{},...b.expiresAt?{expiresAt:b.expiresAt}:{},...(b.agents??[]).length?{agents:b.agents}:{}}));return{pending:[...(s?.pending??[]).map(u=>({id:u.id,kind:"action",title:`${u.connector}:${u.tool}`,requestedBy:u.requestedBy??null,detail:"awaiting confirm"})),...i,...n,...l],onDecide:(u,b,h)=>{u.kind==="action"?s?.resolve(u.id,b==="approve"?"confirm":"deny"):u.kind==="capability"?r(u.id,b==="approve"?"granted":"revoked",h):e(u.id,Ns(b))}}}function Ii(t,e){let r=e?.pending.filter(o=>k(o)&&o.id)??[],s=S(R(t).limit),n=s&&s>0?Math.trunc(s):Ti;return{items:r.slice(0,n),total:r.length}}var Ni=6e4,Mi=10,Mr=new Map;function Ms(t){let e=Mr.get(t);return e||(e={timestamps:[],inFlight:!1},Mr.set(t,e)),e}async function Cs(t){let e=t.now??(()=>Date.now()),r=Ms(t.widgetKey),s=e()-Ni;if(r.timestamps=r.timestamps.filter(n=>n>s),r.inFlight||r.timestamps.length>=Mi)return"rate_limited";r.inFlight=!0;try{return await t.confirmPrompt(t.text)?(r.timestamps.push(e()),await t.sendPrompt(t.text),"sent"):"declined"}finally{r.inFlight=!1}}var Ci=new Set(["health","system-presence","usage.status","usage.cost","agents.list","sessions.list","sessions.resolve","sessions.get","sessions.usage","sessions.usage.timeseries","sessions.usage.logs","node.list","node.describe","cron.get","cron.list","cron.status","cron.runs"]);function Oi(t){return Ci.has(t)}var Bi=new Set(["presence","sessions.changed","boardstate.changed"]);function Os(t){return Bi.has(t)}var Di=1e4,Cr=8*1024,he=256,Li=6e4,Pi=60,Or=new Map;function Wi(t){let e=Or.get(t);return e||(e={timestamps:[]},Or.set(t,e)),e}function Ui(t){let e;try{e=JSON.stringify(t)}catch{return null}return e===void 0?0:typeof TextEncoder<"u"?new TextEncoder().encode(e).length:e.length}var zi=new Set(["dashboard:ready","dashboard:getData","dashboard:getTheme","dashboard:sendPrompt","dashboard:getState","dashboard:setState","dashboard:publish","dashboard:subscribe","dashboard:unsubscribe"]);function ji(t){return typeof t=="object"&&t!==null&&!Array.isArray(t)}function Fi(t){return ji(t)&&t.v===1&&typeof t.type=="string"&&zi.has(t.type)}function Hi(t){let e=t.now??(()=>Date.now()),r=t.getDataTimeoutMs??Di,s=new Set(t.manifest.bindingIds),n=new Set(t.manifest.capabilities),o=0,a=!1,i=Ms(t.manifest.name),l=Wi(t.manifest.name),u=new Map,b=new Set;function h(f,y,A){t.post({v:1,type:"dashboard:error",...A!==void 0?{requestId:A}:{},code:f,message:y})}async function m(f,y){if(!s.has(y)){h("binding_denied",`binding not declared in manifest: ${y}`,f);return}let A=t.assertBindingAllowed?.(y);if(A){h(A,`binding not allowed: ${y}`,f);return}let H=!1,Z=setTimeout(()=>{H||a||(H=!0,b.delete(Z),h("timeout","binding resolution timed out",f))},r);b.add(Z);try{let et=await t.resolveBinding(y);if(H||a)return;H=!0,clearTimeout(Z),b.delete(Z),t.post({v:1,type:"dashboard:data",requestId:f,bindingId:y,data:et})}catch(et){if(H||a)return;H=!0,clearTimeout(Z),b.delete(Z),h("resolve_failed",et instanceof Error?et.message:String(et),f)}}function w(f){t.post({v:1,type:"dashboard:theme",requestId:f,tokens:t.resolveTheme()})}async function v(f,y){if(!n.has("prompt:send")){h("capability_denied","widget lacks the prompt:send capability",f);return}try{let A=await Cs({widgetKey:t.manifest.name,text:y,confirmPrompt:t.confirmPrompt,sendPrompt:t.sendPrompt,now:e});if(a)return;A==="rate_limited"?h("rate_limited","prompt send rate limit exceeded",f):A==="declined"&&h("prompt_declined","operator declined the prompt",f)}catch(A){a||h("resolve_failed",A instanceof Error?A.message:String(A),f)}}async function g(f){if(!n.has("state:persist")||!t.getWidgetState){h("capability_denied","widget lacks the state:persist capability",f);return}try{let y=await t.getWidgetState();if(a)return;t.post({v:1,type:"dashboard:state",requestId:f,state:y.state,...y.version!==void 0?{version:y.version}:{}})}catch(y){a||h("resolve_failed",y instanceof Error?y.message:String(y),f)}}async function _(f,y){if(!n.has("state:persist")||!t.setWidgetState){h("capability_denied","widget lacks the state:persist capability",f);return}try{let{version:A}=await t.setWidgetState(y);if(a)return;t.post({v:1,type:"dashboard:state",requestId:f,state:y,version:A})}catch(A){a||h("resolve_failed",A instanceof Error?A.message:String(A),f)}}function $(f,y,A){if(!n.has("bus:pubsub")){h("capability_denied","widget lacks the bus:pubsub capability",A);return}if(!t.bus)return;let H=Ui(y);if(H===null){h("malformed","publish payload is not serializable",A);return}if(H>Cr){h("payload_too_large",`publish payload exceeds ${Cr} bytes`,A);return}let Z=e()-Li;if(l.timestamps=l.timestamps.filter(et=>et>Z),l.timestamps.length>=Pi){h("rate_limited","publish rate limit exceeded",A);return}l.timestamps.push(e()),t.bus.publish(f,y)}function D(f){if(!n.has("bus:pubsub")||!t.bus){n.has("bus:pubsub")||h("capability_denied","widget lacks the bus:pubsub capability");return}if(u.has(f))return;let y=t.bus.subscribe(f,(A,H)=>{a||t.post({v:1,type:"dashboard:message",channel:A,payload:H})});u.set(f,y)}function I(f){let y=u.get(f);y&&(u.delete(f),y())}function Y(f){if(a)return!1;if(!Fi(f))return o+=1,!1;switch(f.type){case"dashboard:ready":return!0;case"dashboard:getData":{let y=typeof f.requestId=="string"?f.requestId:null,A=typeof f.bindingId=="string"?f.bindingId:null;return y===null||A===null?(o+=1,!1):(m(y,A),!0)}case"dashboard:getTheme":{let y=typeof f.requestId=="string"?f.requestId:null;return y===null?(o+=1,!1):(w(y),!0)}case"dashboard:sendPrompt":{let y=typeof f.requestId=="string"?f.requestId:null,A=typeof f.text=="string"?f.text:null;return y===null||A===null||!A.trim()?(o+=1,!1):(v(y,A),!0)}case"dashboard:getState":{let y=typeof f.requestId=="string"?f.requestId:null;return y===null?(o+=1,!1):(g(y),!0)}case"dashboard:setState":{let y=typeof f.requestId=="string"?f.requestId:null;return y===null||!Object.hasOwn(f,"state")?(o+=1,!1):(_(y,f.state),!0)}case"dashboard:publish":{let y=typeof f.channel=="string"?f.channel:null,A=typeof f.requestId=="string"?f.requestId:void 0;return y===null||!y.trim()||y.length>he||!("payload"in f)?(o+=1,!1):($(y,f.payload,A),!0)}case"dashboard:subscribe":{let y=typeof f.channel=="string"?f.channel:null;return y===null||!y.trim()||y.length>he?(o+=1,!1):(D(y),!0)}case"dashboard:unsubscribe":{let y=typeof f.channel=="string"?f.channel:null;return y===null||!y.trim()||y.length>he?(o+=1,!1):(I(y),!0)}default:return o+=1,!1}}async function Wt(f){if(!(a||!s.has(f)||t.assertBindingAllowed?.(f)))try{let y=await t.resolveBinding(f);a||t.post({v:1,type:"dashboard:push",bindingId:f,data:y})}catch{}}return{handleMessage:Y,push:Wt,get droppedCount(){return o},dispose(){a=!0;for(let f of b)clearTimeout(f);b.clear();for(let f of u.values())f();u.clear(),i.inFlight=!1}}}var Gt=new WeakMap;function Vi(t){let e=Gt.get(t);return e||(e={entries:new Map,self:null,pendingSelfSlug:null},Gt.set(t,e)),e}function qi(t,e){for(let[r,s]of t.entries)s.at+3e4<=e&&t.entries.delete(r)}function Gi(t,e,r=Date.now()){let s=Gt.get(t);return s?(qi(s,r),[...s.entries.entries()].filter(([n,o])=>o.tabSlug===e&&n!==s.self).toSorted((n,o)=>o[1].at-n[1].at).map(([n])=>n)):[]}function Ki(t){Gt.delete(t)}function Br(t,e,r){if(!e)return;let s=Vi(t);s.self===null&&(s.pendingSelfSlug=r),e.request("dashboard.presence.ping",{tabSlug:r}).catch(()=>{})}var Ji="boardstate.changed",Dr=new WeakMap,we=new WeakMap,ve=new WeakMap,_e=new WeakMap,xe=new WeakMap,Xi=45e3,Ot=new WeakMap;function Yi(t,e){Ot.get(t)?.(),Ot.set(t,e)}function Zi(t){Ot.delete(t)}function Qi(t){let e=Ot.get(t);e&&(Ot.delete(t),e())}function td(t){let e=Dr.get(t);return e||(e={loading:!1,loaded:!1,error:null,workspace:null,activeSlug:null,hiddenMenuOpen:!1,pendingWidgetIds:new Set,actionError:null,requestUpdate:null},Dr.set(t,e)),e}function B(t){t.requestUpdate?.()}function ie(t){return!!t&&typeof t=="object"&&!Array.isArray(t)}function ed(t,e=0){return typeof t=="number"&&Number.isFinite(t)?t:e}function q(t){return t instanceof Error&&t.message.trim()?t.message.trim():typeof t=="string"&&t.trim()?t.trim():"Unknown dashboard error."}async function dt(t,e,r){if(e){r?.silent||(t.loading=!0,t.error=null,B(t));try{let s=await e.request("dashboard.workspace.get",{}),n=ws(ie(s)&&"doc"in s?s.doc:s);t.workspace=n,t.activeSlug=_s(n,r?.requestedSlug??t.activeSlug),t.error=null,t.loaded=!0}catch(s){t.error=q(s)}finally{t.loading=!1,B(t)}}}function rd(t,e,r){if(!r){$e(t);return}if(ve.get(t)===r)return;$e(t);let s=r.addEventListener(Ji,n=>{let o=ed((ie(n)?n:void 0)?.workspaceVersion,NaN),a=e.workspace?.workspaceVersion??-1;Number.isFinite(o)&&o<=a||dt(e,r,{silent:!0})});we.set(t,s),ve.set(t,r)}function $e(t){we.get(t)?.(),we.delete(t),ve.delete(t)}function sd(t,e,r,s=Xi){if(!e){Bs(t);return}if(xe.get(t))return;let n=setInterval(()=>{typeof document<"u"&&document.visibilityState==="hidden"||r()},Math.max(1e4,s));_e.set(t,n),xe.set(t,!0)}function Bs(t){let e=_e.get(t);e!==void 0&&(clearInterval(e),_e.delete(t)),xe.delete(t)}function nd(t){Qi(t),$e(t),Bs(t),Ki(t)}function de(t,e,r,s){return{...t,tabs:t.tabs.map(n=>n.slug!==e?n:{...n,widgets:n.widgets.map(o=>o.id===r?s(o):o)})}}function Ds(t,e,r){return{...t,tabs:t.tabs.map(s=>s.slug!==e?s:{...s,widgets:s.widgets.filter(n=>n.id!==r)})}}async function lt(t,e,r){if(!e||!t.workspace)return;let s=t.workspace,n=r.optimistic(s);t.workspace=n,t.pendingWidgetIds.add(r.widgetId),t.actionError=null,B(t);try{await e.request(r.method,r.rpcParams)}catch(o){t.workspace===n&&(t.workspace=s),t.actionError=q(o)}finally{t.pendingWidgetIds.delete(r.widgetId),B(t)}}function Lr(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.move",rpcParams:{tab:r.slug,id:r.widgetId,grid:r.grid},optimistic:s=>de(s,r.slug,r.widgetId,n=>({...n,grid:r.grid}))})}function od(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.update",rpcParams:{tab:r.slug,id:r.widgetId,patch:{collapsed:r.collapsed}},optimistic:s=>de(s,r.slug,r.widgetId,n=>({...n,collapsed:r.collapsed}))})}function ad(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.update",rpcParams:{tab:r.slug,id:r.widgetId,patch:{title:r.title}},optimistic:s=>de(s,r.slug,r.widgetId,n=>({...n,title:r.title}))})}function id(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.update",rpcParams:{tab:r.slug,id:r.widgetId,patch:{ephemeral:null}},optimistic:s=>de(s,r.slug,r.widgetId,n=>{let{ephemeral:o,...a}=n;return a})})}function dd(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.update",rpcParams:{tab:r.slug,id:r.widgetId,patch:{hidden:!0}},optimistic:s=>Ds(s,r.slug,r.widgetId)})}function ld(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.remove",rpcParams:{tab:r.slug,id:r.widgetId},optimistic:s=>Ds(s,r.slug,r.widgetId)})}function cd(t,e,r){return lt(t,e,{widgetId:r.widgetId,method:"dashboard.widget.move",rpcParams:{tab:r.fromSlug,id:r.widgetId,toTab:r.toSlug},optimistic:s=>{let n=s.tabs.find(o=>o.slug===r.fromSlug)?.widgets.find(o=>o.id===r.widgetId);return n?{...s,tabs:s.tabs.map(o=>o.slug===r.fromSlug?{...o,widgets:o.widgets.filter(a=>a.id!==r.widgetId)}:o.slug===r.toSlug?{...o,widgets:[...o.widgets,n]}:o)}:s}})}async function ud(t,e,r){if(!e||!t.workspace)return;let s=t.workspace,n={...s,tabs:s.tabs.map(o=>o.slug===r.slug?{...o,layout:r.layout}:o)};t.workspace=n,t.actionError=null,B(t);try{await e.request("dashboard.tab.update",{slug:r.slug,patch:{layout:r.layout}})}catch(o){t.workspace===n&&(t.workspace=s),t.actionError=q(o),B(t)}}async function bd(t,e){if(e){t.actionError=null,B(t);try{await e.request("dashboard.workspace.undo",{}),await dt(t,e,{silent:!0})}catch(r){t.actionError=q(r),B(t)}}}async function Kt(t,e,r){if(e){t.actionError=null,B(t);try{await e.request("dashboard.widget.approve",{name:r.name,decision:r.decision})}catch(s){t.actionError=q(s),B(t)}}}async function hd(t,e,r){if(e){t.actionError=null,B(t);try{await e.request("dashboard.capability.approve",{name:r.name,decision:r.decision,...r.tools!==void 0?{tools:r.tools}:{},...r.autoConfirm!==void 0?{autoConfirm:r.autoConfirm}:{},...r.expiresAt!==void 0?{expiresAt:r.expiresAt}:{}})}catch(s){t.actionError=q(s),B(t)}}}async function pd(t,e={}){if(!t)throw new Error("Not connected.");let r=ba(await t.request("dashboard.workspace.get",{}));return{filename:ha(),json:ma(r,e)}}async function gd(t,e,r){if(!e)return!1;t.actionError=null,B(t);try{let s=Es(ya(r));return await e.request("dashboard.workspace.replace",{doc:s}),await dt(t,e,{silent:!0}),!0}catch(s){return t.actionError=q(s),B(t),!1}}async function fd(t,e,r){if(!e)return!1;t.actionError=null,B(t);try{let s=xa(r);return await e.request("dashboard.workspace.replace",{doc:s}),await dt(t,e,{silent:!0}),!0}catch(s){return t.actionError=q(s),B(t),!1}}async function ze(t,e){try{if(e.source==="static")return{value:e.value};if(!t)return{error:"Not connected."};if(e.source==="rpc")return e.method?{value:ae(await t.request(e.method,e.params??{}),e.pointer)}:{error:"Binding is missing an rpc method."};if(e.source==="stream")return{error:"Stream bindings resolve via subscription, not a one-shot read."};if(e.source==="computed")return{error:"Computed bindings resolve from sibling values, not a one-shot read."};if(e.source==="mcp")return await md(t,e);let r=await t.request("dashboard.data.read",{binding:e});return{value:ie(r)&&"data"in r?r.data:r}}catch(r){return{error:q(r)}}}async function md(t,e){return!e.connector||!e.tool?{error:"mcp binding is missing a connector or tool."}:{value:ae(yd(await t.request("dashboard.connector.read",{connector:e.connector,tool:e.tool,...e.args?{args:e.args}:{}})),e.pointer)}}function yd(t){if(ie(t)){if("structuredContent"in t&&t.structuredContent!==void 0)return t.structuredContent;if("content"in t)return t.content}return t}function Ls(t,e){if(typeof t=="number"&&Number.isFinite(t))e.push(t);else if(Array.isArray(t))for(let r of t)Ls(r,e)}function wd(t){return Array.isArray(t)?t.length:t==null?0:1}function vd(t,e){return t.replace(/\{(\d+)\}/g,(r,s)=>{let n=e[Number(s)];return typeof n=="string"?n:typeof n=="number"||typeof n=="boolean"||typeof n=="bigint"?String(n):n==null?"":JSON.stringify(n)??""})}function _d(t,e,r){switch(t){case"sum":case"avg":case"min":case"max":{let s=[];for(let n of e)Ls(n,s);return t==="sum"?{value:s.reduce((n,o)=>n+o,0)}:s.length===0?{value:null}:t==="avg"?{value:s.reduce((n,o)=>n+o,0)/s.length}:{value:t==="min"?Math.min(...s):Math.max(...s)}}case"count":return{value:e.reduce((s,n)=>s+wd(n),0)};case"last":return{value:e.length?e[e.length-1]:null};case"pick":return{value:ae(e[0],r)};case"format":return{value:vd(r??"",e)};default:return{error:`Unknown computed op: ${t}`}}}function xd(t,e,r){let s=e.event;return!t||!s||!Os(s)?()=>{}:t.addEventListener(s,n=>{try{r({value:ae(n,e.pointer)})}catch(o){r({error:q(o)})}})}var $d=["--bg","--card","--card-foreground","--text","--muted","--border","--accent","--accent-foreground","--radius","--radius-sm","--font-sans","--font-mono"];function Ad(){let t={};if(typeof document>"u"||typeof getComputedStyle!="function")return t;let e=getComputedStyle(document.documentElement);for(let r of $d){let s=e.getPropertyValue(r).trim();s&&(t[r]=s)}return t}function Pr(t,e){return{get:async()=>{let r=await t.request("dashboard.widget.state.get",{widgetId:e});return{state:r?.state??null,...typeof r?.version=="number"?{version:r.version}:{}}},set:async r=>{let s=(await t.request("dashboard.widget.state.set",{widgetId:e,state:r}))?.version;return{version:typeof s=="number"?s:0}}}}function kd(t,e){let s=Hi({...e,post:a=>{t.contentWindow?.postMessage(a,"*")}}),n=a=>{a.source===t.contentWindow&&s.handleMessage(a.data)},o=t.ownerDocument?.defaultView??(typeof window<"u"?window:null);return o?.addEventListener("message",n),()=>{o?.removeEventListener("message",n),s.dispose()}}var Ed=new Set(["data:read","prompt:send","state:persist","bus:pubsub"]);function Ps(t,e,r){return`${t.replace(/\/+$/,"")}/widgets/${encodeURIComponent(e)}/${r.split("/").map(s=>encodeURIComponent(s)).join("/")}`}async function Td(t,e){if(typeof fetch!="function")return null;try{let r=await fetch(Ps(t,e,"widget.json"),{method:"GET",credentials:"same-origin",headers:{Accept:"application/json"}});if(!r.ok)return null;let s=await r.json();if(typeof s!="object"||s===null)return null;let n=s;return{name:e,bindingIds:(Array.isArray(n.bindings)?n.bindings:[]).map(o=>typeof o=="object"&&o!==null?o.id:void 0).filter(o=>typeof o=="string"),capabilities:(Array.isArray(n.capabilities)?n.capabilities:[]).filter(o=>typeof o=="string"&&Ed.has(o))}}catch{return null}}function Jt(t){return!!t&&typeof t=="object"&&!Array.isArray(t)}function ut(t){return typeof t=="number"&&Number.isFinite(t)?t:0}function Sd(t){if(Jt(t))return{added:ut(t.added),removed:ut(t.removed),moved:ut(t.moved),retitled:ut(t.retitled),tabsChanged:ut(t.tabsChanged),total:ut(t.total)}}async function Rd(t){if(!t)return[];let e=await t.request("dashboard.workspace.history.list",{});return(Jt(e)&&Array.isArray(e.entries)?e.entries:[]).filter(Jt).map(r=>{let s=Sd(r.summary);return{version:typeof r.version=="number"?r.version:0,savedAt:typeof r.savedAt=="string"?r.savedAt:"",bytes:typeof r.bytes=="number"?r.bytes:0,...s?{summary:s}:{}}}).filter(r=>r.version>0)}async function Id(t,e){if(!t)return null;let r=await t.request("dashboard.workspace.history.get",{version:e});return ws(Jt(r)&&"doc"in r?r.doc:r)}async function le(t,e,r){if(typeof fetch!="function")throw new Error("This browser cannot fetch the widget gallery.");let s=await fetch(t,{method:"GET",credentials:"omit",headers:{Accept:"application/json"}});if(!s.ok)throw new Error(`${r} request failed (${s.status}).`);let n=await s.text();if(ka(n)>e)throw new Error(`${r} is too large (max ${Math.floor(e/1024)} KB).`);return n}async function Nd(t){return Ea(await le(t,Ts,"The gallery index"),t)}async function Md(t){return Ra(await le(t,$a,"The widget bundle"))}async function Cd(t){return Na(await le(t,Ts,"The gallery index"),t)}async function Od(t){return Ma(await le(t,Aa,"The recipe bundle"))}async function Bd(t,e){if(!t)throw new Error("Not connected.");await t.request("dashboard.widget.install",{name:e.name,manifest:e.manifest,files:e.files})}var{I:db}=Pn,Dd=t=>t.strings===void 0;var Ws={ATTRIBUTE:1,CHILD:2,PROPERTY:3,BOOLEAN_ATTRIBUTE:4,EVENT:5,ELEMENT:6},je=t=>(...e)=>({_$litDirective$:t,values:e}),Us=class{constructor(t){}get _$AU(){return this._$AM._$AU}_$AT(t,e,r){this._$Ct=t,this._$AM=e,this._$Ci=r}_$AS(t,e){return this.update(t,e)}update(t,e){return this.render(...e)}};var Rt=(t,e)=>{let r=t._$AN;if(r===void 0)return!1;for(let s of r)s._$AO?.(e,!1),Rt(s,e);return!0},Xt=t=>{let e,r;do{if((e=t._$AM)===void 0)break;r=e._$AN,r.delete(t),t=e}while(r?.size===0)},zs=t=>{for(let e;e=t._$AM;t=e){let r=e._$AN;if(r===void 0)e._$AN=r=new Set;else if(r.has(t))break;r.add(t),Wd(e)}};function Ld(t){this._$AN!==void 0?(Xt(this),this._$AM=t,zs(this)):this._$AM=t}function Pd(t,e=!1,r=0){let s=this._$AH,n=this._$AN;if(n!==void 0&&n.size!==0)if(e)if(Array.isArray(s))for(let o=r;o<s.length;o++)Rt(s[o],!1),Xt(s[o]);else s!=null&&(Rt(s,!1),Xt(s));else Rt(this,t)}var Wd=t=>{t.type==Ws.CHILD&&(t._$AP??(t._$AP=Pd),t._$AQ??(t._$AQ=Ld))},js=class extends Us{constructor(){super(...arguments),this._$AN=void 0}_$AT(t,e,r){super._$AT(t,e,r),zs(this),this.isConnected=t._$AU}_$AO(t,e=!0){t!==this.isConnected&&(this.isConnected=t,t?this.reconnected?.():this.disconnected?.()),e&&(Rt(this,t),Xt(this))}setValue(t){if(Dd(this._$Ct))this._$Ct._$AI(t,this);else{let e=[...this._$Ct._$AH];e[this._$Ci]=t,this._$Ct._$AI(e,this,0)}}disconnected(){}reconnected(){}};function Wr(t,e){return t.bindings?.[e]??null}function Ud(t){let{iframe:e,widget:r,manifest:s,context:n}=t,o=n.tabSlug??"",a=ea(),i=kd(e,{manifest:s,bus:{publish:(l,u)=>oa({tabSlug:o,channel:l,fromSubscriberId:a,payload:u}),subscribe:(l,u)=>ra({tabSlug:o,channel:l,subscriberId:a,deliver:u})},getWidgetState:async()=>{if(!n.transport)throw new Error("Not connected.");return Pr(n.transport,r.id).get()},setWidgetState:async l=>{if(!n.transport)throw new Error("Not connected.");return Pr(n.transport,r.id).set(l)},assertBindingAllowed:l=>{let u=Wr(r,l);return u?.source==="rpc"&&!Oi(u.method??"")||u?.source==="stream"&&!Os(u.event??"")?"binding_denied":null},resolveBinding:async l=>{let u=Wr(r,l);if(!u)throw new Error(`binding not configured: ${l}`);let b=await ze(n.transport,u);if("error"in b)throw new Error(b.error);return b.value},resolveTheme:n.readThemeTokens??Ad,confirmPrompt:async l=>n.confirmPrompt?await n.confirmPrompt(l):typeof window<"u"?window.confirm(l):!1,sendPrompt:async l=>{if(!n.transport)throw new Error("Not connected.");await n.transport.request("chat.send",{sessionKey:n.sessionKey,message:l,deliver:!1})}});return()=>{i(),na(o,a)}}var zd=class extends js{constructor(...t){super(...t),this.iframe=null,this.detach=null,this.key=""}render(t){let e=t.widget.kind.slice(7),r=Ps(t.context.basePath,e,"index.html"),s=`${t.widget.id}::${r}`;if(this.iframe&&this.key===s)return this.iframe;this.detach?.();let n=document.createElement("iframe");return n.setAttribute("sandbox","allow-scripts"),n.setAttribute("referrerpolicy","no-referrer"),n.setAttribute("loading","lazy"),n.className="dashboard-widget__frame",n.title=t.widget.title,n.src=r,n.setAttribute("data-test-id","boardstate-custom-widget-frame"),this.detach=Ud({iframe:n,widget:t.widget,manifest:t.manifest,context:t.context}),this.iframe=n,this.key=s,n}disconnected(){this.detach?.(),this.detach=null,this.iframe=null,this.key=""}},jd=je(zd);function Fd(t){return c`<div class="dashboard-widget__custom" data-test-id="boardstate-custom-widget">
    ${jd(t)}
  </div>`}function L(t){return c`<svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    ${t}
  </svg>`}var M={spark:L(x`<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2 2M16 16l2 2M18 6l-2 2M8 16l-2 2" />`),x:L(x`<path d="M18 6L6 18M6 6l12 12" />`),plus:L(x`<path d="M12 5v14M5 12h14" />`),eyeOff:L(x`<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20C5 20 1 12 1 12a18.5 18.5 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M1 1l22 22" />`),chevronRight:L(x`<path d="M9 18l6-6-6-6" />`),chevronDown:L(x`<path d="M6 9l6 6 6-6" />`),arrowUpDown:L(x`<path d="M7 15l5 5 5-5M7 9l5-5 5 5" />`),moreHorizontal:L(x`<circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" />`),externalLink:L(x`<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><path d="M15 3h6v6" /><path d="M10 14L21 3" />`),clock:L(x`<circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />`),puzzle:L(x`<path d="M4 7h3a1.5 1.5 0 1 0 3 0h3v3a1.5 1.5 0 1 1 0 3v3h-3a1.5 1.5 0 1 0-3 0H4v-3a1.5 1.5 0 1 1 0-3z" />`),maximize:L(x`<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3" />`),minimize:L(x`<path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3" />`)};var Ur=()=>new Hd,Hd=class{},pe=new WeakMap,Bt=je(class extends js{render(t){return p}update(t,[e]){let r=e!==this.G;return r&&this.rt(void 0),(r||this.lt!==this.ct)&&(this.G=e,this.ht=t.options?.host,this.rt(this.ct=t.element)),p}rt(t){if(this.G!==void 0)if(this.isConnected||(t=void 0),typeof this.G=="function"){let e=this.ht??globalThis,r=pe.get(e);r===void 0&&(r=new WeakMap,pe.set(e,r)),r.get(this.G)!==void 0&&this.G.call(this.ht,void 0),r.set(this.G,t),t!==void 0&&this.G.call(this.ht,t)}else this.G.value=t}get lt(){return typeof this.G=="function"?pe.get(this.ht??globalThis)?.get(this.G):this.G?.value}disconnected(){this.lt===this.ct&&this.rt(void 0)}reconnected(){this.rt(this.ct)}}),Yt={"common.save":"Save","common.cancel":"Cancel","common.reload":"Reload","common.loading":"Loading\u2026","common.dismiss":"Dismiss","dashboard.header.subtitle":"Your pinned widgets and workspaces.","dashboard.tabs.label":"Workspaces","dashboard.tabs.hidden":"Hidden ({count})","dashboard.error.title":"Couldn\u2019t load your workspace","dashboard.error.subtitle":"Something went wrong reading the workspace document.","dashboard.error.detailSummary":"Error detail","dashboard.empty.onboardingTitle":"No workspaces yet","dashboard.empty.onboardingSubtitle":"Ask the agent to add a workspace tab, or use the CLI.","dashboard.empty.onboardingCommand":"boardstate tab add <name>","dashboard.empty.noVisibleTabs":"All workspace tabs are hidden.","dashboard.empty.tabTitle":"This workspace is empty","dashboard.empty.tabSubtitle":"Ask the agent to add a widget here.","dashboard.onboarding.title":"Add your first workspace","dashboard.onboarding.primary":"Ask the agent to create a workspace tab for you.","dashboard.onboarding.secondary":"Or add one from the CLI:","dashboard.widget.editTitleTitle":"Edit widget title","dashboard.widget.editTitleLabel":"Widget title","dashboard.widget.moveToTabTitle":"Move widget to tab","dashboard.widget.moveToTabEmpty":"There are no other tabs to move this widget to.","dashboard.widget.menu.editTitle":"Edit title","dashboard.widget.menu.moveToTab":"Move to tab","dashboard.widget.menu.hide":"Hide","dashboard.widget.menu.remove":"Remove","dashboard.widget.provenanceChip":"AI","dashboard.widget.provenanceTooltip":"Created by {agent}","dashboard.widget.agentChipTooltip":"Built by {agent}","dashboard.widget.expand":"Expand widget","dashboard.widget.collapse":"Collapse widget","dashboard.widget.moveHandle":"Move widget","dashboard.widget.resizeHandle":"Resize widget","dashboard.widget.menuLabel":"Widget menu","dashboard.widget.errorTitle":"This widget hit an error","dashboard.widget.errorHumane":"The rest of your workspace is unaffected.","dashboard.widget.errorDetailSummary":"Error detail","dashboard.widget.customPlaceholder":"Custom widget","dashboard.widget.customLoading":"Loading widget\u2026","dashboard.widget.unknownKind":"Unknown widget: {kind}","dashboard.widget.approval.title":"Approve this widget?","dashboard.widget.approval.byAgent":"Requested by {agent}","dashboard.widget.approval.byUnknown":"Requested by an agent","dashboard.widget.approval.approve":"Approve","dashboard.widget.approval.reject":"Reject","dashboard.widget.approval.unavailable":"This widget is unavailable.","dashboard.widget.stat.empty":"\u2014","dashboard.widget.markdownEmpty":"Nothing to show yet.","dashboard.widget.table.empty":"No rows to show.","dashboard.widget.table.more":"+{count} more","dashboard.widget.sessions.empty":"No sessions yet.","dashboard.widget.usage.cost":"Cost","dashboard.widget.usage.tokens":"Tokens","dashboard.widget.cron.empty":"No scheduled jobs.","dashboard.widget.cron.next":"Next {time}","dashboard.widget.cron.noNext":"Not scheduled","dashboard.widget.instances.empty":"No connected instances.","dashboard.widget.instances.idle":"idle {duration}","dashboard.widget.activity.empty":"No recent activity.","dashboard.widget.embed.missing":"No URL configured for this embed.","dashboard.widget.embed.blockedExternal":"External embeds are blocked by policy.","dashboard.widget.embed.blockedScheme":"This URL scheme cannot be embedded.","dashboard.widget.chart.empty":"No data to chart.","dashboard.widget.chart.label":"Chart","dashboard.widget.notes.placeholder":"Write a note\u2026","dashboard.widget.notes.readonlyHint":"Connect to the gateway to edit and save notes.","dashboard.widget.actionForm.empty":"This action form has no fields yet.","dashboard.widget.actionForm.submit":"Send","dashboard.widget.actionForm.toolPending":"Submitted \u2014 waiting for operator confirmation.","dashboard.widget.actionButton.run":"Run","dashboard.widget.actionButton.invoking":"Invoking\u2026","dashboard.widget.actionButton.pending":"Waiting for operator confirmation\u2026","dashboard.widget.actionButton.confirm":"Confirm","dashboard.widget.actionButton.deny":"Deny","dashboard.widget.actionButton.operatorOnly":"Only the local operator can confirm this action.","dashboard.widget.actionButton.confirmed":"Confirmed.","dashboard.widget.actionButton.denied":"Denied by the operator.","dashboard.widget.actionButton.expired":"The confirmation window expired.","dashboard.widget.actionButton.resultLabel":"Result","dashboard.widget.actionButton.errorLabel":"Error","dashboard.widget.actionButton.disconnected":"Connect to the gateway to run this action.","dashboard.widget.actionButton.misconfigured":"This action is missing a connector or tool.","dashboard.widget.preview.missing":"This preview has no URL yet.","dashboard.widget.preview.blockedExternal":"External previews are disabled by your gateway policy.","dashboard.widget.preview.blockedScheme":"This preview URL uses an unsupported scheme.","dashboard.widget.preview.reload":"Reload preview","dashboard.widget.preview.viewport.desktop":"Desktop","dashboard.widget.preview.viewport.tablet":"Tablet","dashboard.widget.preview.viewport.mobile":"Mobile","dashboard.widget.agentStatus.empty":"No agents yet.","dashboard.widget.agentStatus.busy":"Busy","dashboard.widget.agentStatus.idle":"Idle","dashboard.widget.agentStatus.progress":"{percent}% of budget","dashboard.widget.approvals.empty":"No pending approvals.","dashboard.widget.approvals.approve":"Approve","dashboard.widget.approvals.deny":"Deny","dashboard.widget.approvals.confirm":"Confirm","dashboard.widget.approvals.requestedBy":"Requested by {agent}","dashboard.widget.approvals.kind.widget":"Widget","dashboard.widget.approvals.kind.capability":"Data source","dashboard.widget.approvals.kind.action":"Action","dashboard.widget.approvals.autoConfirm":"Auto-run","dashboard.widget.approvals.autoConfirmHint":"Runs without confirmation each time","dashboard.widget.approvals.scopeLabel":"Agents","dashboard.widget.approvals.scopeAll":"All agents","dashboard.widget.approvals.scopedTo":"Scoped to {agents}","dashboard.widget.approvals.ttlLabel":"Expires in (min)","dashboard.widget.approvals.expiresIn":"Expires in {duration}","dashboard.widget.approvals.expiresSoon":"Expiring\u2026","dashboard.widget.approvals.save":"Save","dashboard.widget.approvals.revoke":"Revoke","dashboard.widget.chat.empty":"Ask the agent to build or change this board\u2026","dashboard.widget.chat.placeholder":"Message the agent\u2026","dashboard.widget.chat.send":"Send","dashboard.widget.chat.stop":"Stop","dashboard.widget.chat.disconnected":"Connect to the gateway to chat with the agent.","dashboard.widget.chat.roleUser":"You","dashboard.widget.chat.roleAssistant":"Agent","dashboard.widget.chat.actionsOne":"1 action","dashboard.widget.chat.actionsMany":"{count} actions","dashboard.widget.chat.building":"building\u2026","dashboard.widget.chat.retrying":"retrying\u2026","dashboard.widget.chat.jumpToLatest":"Jump to latest","dashboard.widget.chat.args":"Arguments","dashboard.widget.chat.result":"Result","dashboard.widget.chat.tool.readBoard":"Read the board","dashboard.widget.chat.tool.createdTab":"Created tab {name}","dashboard.widget.chat.tool.addedWidget":"Added widget {id}","dashboard.widget.chat.approveTitle":"The agent scaffolded widget \u201C{name}\u201D","dashboard.widget.chat.approve":"Approve","dashboard.widget.chat.reject":"Reject","common.close":"Close","common.back":"Back","dashboard.tabs.presence":"{count} viewing","dashboard.tabs.private":"Private \u2014 only you can see this tab","dashboard.tabs.groupUser":"You","dashboard.tabs.groupSystem":"System","dashboard.tabs.groupAgent":"{agent}","dashboard.tabs.collapseGroup":"Collapse {group} tabs","dashboard.tabs.expandGroup":"Expand {group} tabs","dashboard.header.fullBleedEnter":"Full-bleed","dashboard.header.fullBleedExit":"Exit full-bleed","dashboard.agentFilter.label":"Agents","dashboard.agentFilter.all":"All","dashboard.widget.ephemeralBadge":"Temporary","dashboard.widget.ephemeralTooltip":"Temporary answer \u2014 pin it to keep it here.","dashboard.widget.menu.pin":"Pin","dashboard.widget.blame.createdBy":"Created by {actor}","dashboard.widget.blame.createdByVersion":"Created by {actor} \xB7 v{version}","dashboard.widget.blame.logbookLink":"View in logbook","dashboard.history.open":"History","dashboard.history.title":"Workspace history","dashboard.history.subtitle":"Review recent changes, compare against now, and undo the last one.","dashboard.history.empty":"No history yet \u2014 changes appear here after your first edit.","dashboard.history.emptyDetail":"Select a version to preview it.","dashboard.history.version":"Version {version}","dashboard.history.latest":"Latest change","dashboard.history.previewTitle":"Snapshot","dashboard.history.previewEmpty":"This tab had no widgets at this point.","dashboard.history.diffTitle":"Changes since this version","dashboard.history.diffEmpty":"Nothing changed since this version.","dashboard.history.restore":"Undo last change","dashboard.history.restoreConfirm":"Undo the most recent change?","dashboard.history.restoreOnlyNewest":"Only the most recent change can be undone.","dashboard.history.actorUnknown":"Unknown","dashboard.history.kind.widget-added":"Added","dashboard.history.kind.widget-removed":"Removed","dashboard.history.kind.widget-moved":"Moved","dashboard.history.kind.widget-retitled":"Retitled","dashboard.history.kind.tab-added":"Tab added","dashboard.history.kind.tab-removed":"Tab removed","dashboard.history.kind.tab-retitled":"Tab retitled","dashboard.history.summary.added":"+{count}","dashboard.history.summary.removed":"\u2212{count}","dashboard.history.summary.moved":"{count} moved","dashboard.history.summary.retitled":"{count} renamed","dashboard.history.summary.tabs":"{count} tabs","dashboard.history.summary.minor":"Other edit","dashboard.history.previewCaption":"Layout at version {version}","dashboard.gallery.open":"Widget gallery","dashboard.gallery.title":"Widget gallery","dashboard.gallery.subtitle":"Browse a widget registry and install a widget from its URL.","dashboard.gallery.urlLabel":"Registry index URL","dashboard.gallery.urlPlaceholder":"https://example.com/widgets/index.json","dashboard.gallery.browse":"Browse","dashboard.gallery.view":"View","dashboard.gallery.install":"Install","dashboard.gallery.empty":"No widgets found at this registry.","dashboard.gallery.capabilities":"Requested capabilities","dashboard.gallery.noCapabilities":"No special capabilities requested.","dashboard.gallery.pendingNote":"Installed widgets stay pending until you approve them, then run sandboxed.","dashboard.gallery.tabWidgets":"Widgets","dashboard.gallery.tabTemplates":"Templates","dashboard.gallery.recipesEmpty":"No templates found at this registry.","dashboard.gallery.recipeNeedsNothing":"Works out of the box \u2014 no grants required.","dashboard.gallery.recipeNeedsConnectors":"Needs: {connectors}","dashboard.gallery.recipeNeedsLabel":"This board will ask for these tools","dashboard.gallery.recipeNoGrants":"No external tools \u2014 installs ready to use.","dashboard.gallery.recipeReadOnly":"read-only","dashboard.gallery.recipeInstall":"Install template","dashboard.gallery.recipeInstallNote":"Installing imports the board with its grants requested \u2014 approve them in the approvals widget to light it up.","dashboard.distribution.export":"Export","dashboard.distribution.exportTitle":"Download this workspace as a JSON file","dashboard.distribution.import":"Import","dashboard.distribution.importTitle":"Import a workspace from a JSON file"};function Vd(t,e){return e?t.replace(/\{(\w+)\}/g,(r,s)=>Object.hasOwn(e,s)?e[s]:r):t}var Fs={...Yt};function qd(t){Fs=t?{...Yt,...t}:{...Yt}}function d(t,e){return Vd(Fs[t]??Yt[t]??t,e)}function Gd(t){if(t===void 0)return"";if(typeof t=="string")return t;try{return JSON.stringify(t,null,2)}catch{return String(t)}}var Kd=class{constructor(t){this.widgetId=t,this.root=null,this.ctx=null,this.widget=null,this.phase={kind:"idle"},this.unsubscribe=null,this.rootRef=e=>{e instanceof HTMLElement?this.mount(e):this.destroy()},this.onInvoke=()=>{let e=this.ctx?.actions;if(!e||!this.widget)return;let r=Nr(this.widget);if(!r.connector||!r.tool){this.setPhase({kind:"error",message:d("dashboard.widget.actionButton.misconfigured")});return}this.setPhase({kind:"running"}),e.invoke({connector:r.connector,tool:r.tool,...r.args?{args:r.args}:{}}).then(s=>{this.setPhase(s.kind==="pending"?{kind:"pending",id:s.id,expiresAt:s.expiresAt}:{kind:"result",value:s.result})}).catch(s=>{this.setPhase({kind:"error",message:s instanceof Error?s.message:String(s)})})},this.onConfirm=e=>{let r=this.ctx?.actions?.confirm;r&&(this.setPhase({kind:"running"}),r(e).then(({result:s})=>this.setPhase({kind:"result",value:s})).catch(s=>{this.setPhase({kind:"error",message:s instanceof Error?s.message:String(s)})}))},this.onDeny=e=>{let r=this.ctx?.actions?.deny;r&&r(e).then(()=>this.setPhase({kind:"denied"})).catch(s=>{this.setPhase({kind:"error",message:s instanceof Error?s.message:String(s)})})}}setContext(t,e){this.ctx=t,this.widget=e,this.root&&this.renderIsland()}mount(t){this.root=t,this.unsubscribe?.(),this.unsubscribe=null,this.phase={kind:"idle"},this.renderIsland();let e=this.ctx?.actions;e&&(this.unsubscribe=e.subscribe(r=>this.onActionChange(r)))}destroy(){this.unsubscribe?.(),this.unsubscribe=null,this.root=null,Ae.delete(this.widgetId)}onActionChange(t){if(!(this.phase.kind!=="pending"||t.id!==this.phase.id)){if(t.status==="confirmed")this.phase={kind:"confirmed"};else if(t.status==="denied")this.phase={kind:"denied"};else if(t.status==="expired")this.phase={kind:"expired"};else return;this.renderIsland()}}setPhase(t){this.phase=t,this.renderIsland()}renderIsland(){this.root&&Ce(this.template(),this.root)}template(){let t=this.ctx?.actions,e=(this.widget?Nr(this.widget):null)?.label??d("dashboard.widget.actionButton.run"),r=this.phase.kind==="running"||this.phase.kind==="pending";return c`
      <div class="dashboard-action-button" data-test-id="dashboard-action-button">
        <button
          class="bs-btn bs-btn--small bs-btn--primary dashboard-action-button__invoke"
          type="button"
          data-test-id="dashboard-action-button-invoke"
          ?disabled=${!t||r}
          @click=${this.onInvoke}
        >
          ${e}
        </button>
        ${t?this.renderStatus():c`<div
                class="dashboard-action-button__hint"
                data-test-id="dashboard-action-button-disconnected"
              >
                ${d("dashboard.widget.actionButton.disconnected")}
              </div>`}
      </div>
    `}renderStatus(){switch(this.phase.kind){case"idle":return p;case"running":return c`<div class="dashboard-action-button__status" data-status="running">
          ${d("dashboard.widget.actionButton.invoking")}
        </div>`;case"pending":return this.renderPending(this.phase.id);case"confirmed":return c`<div
          class="dashboard-action-button__status"
          data-status="confirmed"
          data-test-id="dashboard-action-button-confirmed"
        >
          ${d("dashboard.widget.actionButton.confirmed")}
        </div>`;case"denied":return c`<div
          class="dashboard-action-button__status"
          data-status="denied"
          data-test-id="dashboard-action-button-denied"
        >
          ${d("dashboard.widget.actionButton.denied")}
        </div>`;case"expired":return c`<div
          class="dashboard-action-button__status"
          data-status="expired"
          data-test-id="dashboard-action-button-expired"
        >
          ${d("dashboard.widget.actionButton.expired")}
        </div>`;case"result":return c`<div class="dashboard-action-button__result" data-status="result">
          <div class="dashboard-action-button__result-label">
            ${d("dashboard.widget.actionButton.resultLabel")}
          </div>
          <pre
            class="dashboard-action-button__result-body"
            data-test-id="dashboard-action-button-result"
          >
${Gd(this.phase.value)}</pre>
        </div>`;case"error":return c`<div
          class="dashboard-action-button__error"
          role="alert"
          data-test-id="dashboard-action-button-error"
        >
          <span class="dashboard-action-button__result-label"
            >${d("dashboard.widget.actionButton.errorLabel")}</span
          >
          <span class="dashboard-action-button__error-message">${this.phase.message}</span>
        </div>`}}renderPending(t){let e=!!(this.ctx?.actions?.confirm&&this.ctx?.actions?.deny);return c`
      <div
        class="dashboard-action-button__pending"
        data-status="pending"
        data-test-id="dashboard-action-button-pending"
      >
        <span class="dashboard-action-button__status-text"
          >${d("dashboard.widget.actionButton.pending")}</span
        >
        ${e?c`<span class="dashboard-action-button__pending-actions">
                <button
                  class="bs-btn bs-btn--small bs-btn--primary"
                  type="button"
                  data-test-id="dashboard-action-button-confirm"
                  @click=${()=>this.onConfirm(t)}
                >
                  ${d("dashboard.widget.actionButton.confirm")}
                </button>
                <button
                  class="bs-btn bs-btn--small"
                  type="button"
                  data-test-id="dashboard-action-button-deny"
                  @click=${()=>this.onDeny(t)}
                >
                  ${d("dashboard.widget.actionButton.deny")}
                </button>
              </span>`:c`<span
                class="dashboard-action-button__operator-only"
                data-test-id="dashboard-action-button-operator-only"
                >${d("dashboard.widget.actionButton.operatorOnly")}</span
              >`}
      </div>
    `}},Ae=new Map;function Jd(t,e,r){let s=Ae.get(t.id);return s||(s=new Kd(t.id),Ae.set(t.id,s)),s.setContext(r,t),c`<div class="dashboard-action-button-host" ${Bt(s.rootRef)}></div>`}function Xd(t){let e=t.type==="select"?c`<select class="dashboard-action-form__control" name=${t.name}>
          ${(t.options??[]).map(r=>c`<option value=${r}>${r}</option>`)}
        </select>`:c`<input
          class="dashboard-action-form__control"
          type=${t.type==="number"?"number":"text"}
          name=${t.name}
          maxlength=${t.maxLength??200}
        />`;return c`<label class="dashboard-action-form__field">
    <span class="dashboard-action-form__label">${t.label}</span>
    ${e}
  </label>`}function Yd(t,e,r,s,n){if(!s.actions||!t.connector||!t.tool)return;let o=mi(t,r);s.actions.invoke({connector:t.connector,tool:t.tool,args:o}).then(a=>{a.kind==="pending"&&s.onActionError?.(d("dashboard.widget.actionForm.toolPending")),n.reset()}).catch(a=>{s.onActionError?.(a instanceof Error?a.message:String(a))})}function Zd(t,e,r){let s=gi(t);if(s.fields.length===0||!s.template)return c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.actionForm.empty")}
    </div>`;let n=a=>{let i={};for(let l of s.fields){let u=a.elements.namedItem(l.name);i[l.name]=u&&"value"in u?String(u.value??""):""}return i};return c`
    <form class="dashboard-action-form" data-test-id="dashboard-action-form" @submit=${a=>{a.preventDefault();let i=a.currentTarget,l=n(i);if(s.mode==="tool"){Yd(s,t,l,r,i);return}let u=fi(s,l);!u.trim()||!r.dispatchPrompt||r.dispatchPrompt({widgetKey:`builtin:action-form:${t.id}`,text:u}).then(b=>{b==="sent"&&i.reset()}).catch(b=>{r.onActionError?.(b instanceof Error?b.message:String(b))})}}>
      ${s.fields.map(Xd)}
      <button
        class="bs-btn bs-btn--small bs-btn--primary dashboard-action-form__submit"
        type="submit"
      >
        ${s.buttonLabel??d("dashboard.widget.actionForm.submit")}
      </button>
    </form>
    ${(s.mode==="tool"?r.actions:r.dispatchPrompt)?p:c`<span hidden data-test-id="dashboard-action-form-inert"></span>`}
  `}function Qd(t){let e=Number.isFinite(t)?t:0;return new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(e)}function tl(t){let e=Number.isFinite(t)?t:0;return new Intl.NumberFormat("en-US",{notation:"compact",maximumFractionDigits:1}).format(e)}function Fe(t){if(!Number.isFinite(t))return"";try{return new Intl.DateTimeFormat("en-US",{month:"short",day:"numeric",hour:"numeric",minute:"2-digit"}).format(new Date(t))}catch{return new Date(t).toISOString()}}function el(t){if(!Number.isFinite(t)||t<0)return"";let e=Math.round(t/1e3);if(e<60)return`${e}s`;let r=Math.floor(e/60),s=e%60;if(r<60)return s?`${r}m ${s}s`:`${r}m`;let n=Math.floor(r/60),o=r%60;return o?`${n}h ${o}m`:`${n}h`}function rl(t){return t==="ok"?"dashboard-badge--ok":t==="error"?"dashboard-badge--error":"dashboard-badge--muted"}function sl(t,e){let r=si(t,e);return r.entries.length===0?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.activity.empty")}
    </div>`:c`
    <ul class="dashboard-feed" data-test-id="dashboard-activity">
      ${r.entries.map(s=>c`
          <li class="dashboard-feed__row">
            <div class="dashboard-feed__head">
              <span class="dashboard-feed__title">${s.title}</span>
              ${s.status?c`<span class="dashboard-badge ${rl(s.status)}"
                      >${s.status}</span
                    >`:p}
              ${s.ts!==null?c`<span class="dashboard-feed__time">${Fe(s.ts)}</span>`:p}
            </div>
            ${s.detail?c`<div class="dashboard-feed__detail">${s.detail}</div>`:p}
          </li>
        `)}
    </ul>
  `}function nl(t,e){let r=Ei(t,e);return r.rows.length===0?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.agentStatus.empty")}
    </div>`:c`
    <ul class="dashboard-list dashboard-agent-status" data-test-id="dashboard-agent-status">
      ${r.rows.map(s=>c`
          <li class="dashboard-list__row">
            <span
              class="dashboard-dot ${s.active?"dashboard-dot--live":""}"
              aria-hidden="true"
            ></span>
            <span class="dashboard-list__label">${s.label}</span>
            <span
              class="dashboard-badge ${s.active?"dashboard-badge--ok":"dashboard-badge--muted"}"
            >
              ${s.active?d("dashboard.widget.agentStatus.busy"):d("dashboard.widget.agentStatus.idle")}
            </span>
            ${s.task?c`<span class="dashboard-list__meta">${s.task}</span>`:p}
            ${s.progress!==null?c`<span class="dashboard-list__meta"
                    >${d("dashboard.widget.agentStatus.progress",{percent:String(Math.round(s.progress*100))})}</span
                  >`:p}
          </li>
        `)}
    </ul>
  `}function ol(t){return d(t==="capability"?"dashboard.widget.approvals.kind.capability":t==="action"?"dashboard.widget.approvals.kind.action":"dashboard.widget.approvals.kind.widget")}function zr(t,e){let r=t.currentTarget?.closest("li");return r?[...r.querySelectorAll(e)].filter(s=>s.checked).map(s=>s.value):[]}function al(t){let e=t.currentTarget?.closest("li")?.querySelector("input.dashboard-approvals__ttl"),r=e&&e.value.trim()!==""?Number(e.value):NaN;if(!(!Number.isFinite(r)||r<=0))return new Date(Date.now()+r*6e4).toISOString()}function il(t,e){let r=al(t);if(!e)return r!==void 0?{expiresAt:r}:{};let s=zr(t,"input.dashboard-approvals__grant"),n=zr(t,"input.dashboard-approvals__auto");return{tools:s,...n.length?{autoConfirm:n.filter(o=>s.includes(o))}:{},...r!==void 0?{expiresAt:r}:{}}}function dl(t){let e=Date.parse(t)-Date.now();if(Number.isNaN(e)||e<=0)return d("dashboard.widget.approvals.expiresSoon");let r=Math.round(e/6e4),s=Math.floor(r/60);return d("dashboard.widget.approvals.expiresIn",{duration:s>0?`${s}h ${r%60}m`:`${r}m`})}function ll(t){let e=t.agents??[],r=e.length>0?d("dashboard.widget.approvals.scopedTo",{agents:e.join(", ")}):d("dashboard.widget.approvals.scopeAll");return c`<span
    class="dashboard-approvals__scope"
    data-test-id="dashboard-approvals-scope"
    data-agents=${e.join(",")}
    >${d("dashboard.widget.approvals.scopeLabel")}: ${r}</span
  >`}function cl(t){let e=t.tools??[],r=new Set(t.autoConfirm??[]);return c`<ul class="dashboard-approvals__tools" data-test-id="dashboard-approvals-tools">
    ${e.map(s=>c`<li>
          <label class="dashboard-approvals__grant-label"
            ><input type="checkbox" class="dashboard-approvals__grant" value=${s} checked /><span
              >${s}</span
            ></label
          >
          <label
            class="dashboard-approvals__auto-label"
            title=${d("dashboard.widget.approvals.autoConfirmHint")}
            ><input
              type="checkbox"
              class="dashboard-approvals__auto"
              value=${s}
              ?checked=${r.has(s)}
            /><span>${d("dashboard.widget.approvals.autoConfirm")}</span></label
          >
        </li>`)}
  </ul>`}function ul(t,e,r){let s=r.approvals,n=Ii(t,s);return n.items.length===0?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.approvals.empty")}
    </div>`:c`
    <ul class="dashboard-list dashboard-approvals" data-test-id="dashboard-approvals">
      ${n.items.map(o=>{let a=o.kind==="capability",i=a&&(o.tools??[]).length>0,l=o.granted?d("dashboard.widget.approvals.save"):o.kind==="action"?d("dashboard.widget.approvals.confirm"):d("dashboard.widget.approvals.approve"),u=h=>{if(!a){s?.onDecide(o,"approve");return}let m=il(h,i);Object.keys(m).length>0?s?.onDecide(o,"approve",m):s?.onDecide(o,"approve")},b=o.granted?d("dashboard.widget.approvals.revoke"):d("dashboard.widget.approvals.deny");return c`
          <li
            class="dashboard-list__row ${o.granted?"dashboard-approvals__row--granted":""}"
          >
            <span class="dashboard-badge dashboard-badge--muted">${ol(o.kind)}</span>
            <span class="dashboard-list__label">${o.title}</span>
            ${o.detail?c`<span class="dashboard-list__meta">${o.detail}</span>`:o.requestedBy?c`<span class="dashboard-list__meta"
                      >${d("dashboard.widget.approvals.requestedBy",{agent:o.requestedBy})}</span
                    >`:p}
            ${o.expiresAt?c`<span
                    class="dashboard-approvals__countdown"
                    data-test-id="dashboard-approvals-countdown"
                    >${dl(o.expiresAt)}</span
                  >`:p}
            ${i?cl(o):p}
            ${a?ll(o):p}
            ${a?c`<label class="dashboard-approvals__ttl-label"
                    >${d("dashboard.widget.approvals.ttlLabel")}
                    <input
                      type="number"
                      min="1"
                      class="dashboard-approvals__ttl"
                      data-test-id="dashboard-approvals-ttl"
                  /></label>`:p}
            <span class="dashboard-approvals__actions">
              <button
                class="bs-btn bs-btn--small bs-btn--primary"
                type="button"
                data-test-id="dashboard-approvals-approve"
                @click=${u}
              >
                ${l}
              </button>
              <button
                class="bs-btn bs-btn--small"
                type="button"
                data-test-id="dashboard-approvals-deny"
                @click=${()=>s?.onDecide(o,"reject")}
              >
                ${b}
              </button>
            </span>
          </li>
        `})}
    </ul>
  `}var J=100,j=40,E=2;function ce(t,e,r){let s=r-e;if(s<=0)return j/2;let n=(t-e)/s;return j-E-n*(j-E*2)}function Dt(t,e){return e<=1?J/2:E+t/(e-1)*(J-E*2)}function He(t,e,r){return t.map((s,n)=>`${Dt(n,t.length)},${ce(s,e,r)}`).join(" ")}var bl=new Intl.NumberFormat("en-US",{notation:"compact",maximumFractionDigits:1});function yt(t){return Number.isFinite(t)?bl.format(t):""}function Hs(t){if(t.length<2)return"flat";let e=t[0],r=t[t.length-1];return r>e?"up":r<e?"down":"flat"}function hl(t){return x`<polyline
    class="dashboard-chart__line"
    fill="none"
    points=${He(t.values,t.min,t.max)}
  />`}function pl(t){let e=He(t.values,t.min,t.max),r=Dt(0,t.values.length),s=Dt(t.values.length-1,t.values.length),n=j-E;return x`<g>
    <polygon class="dashboard-chart__area" points=${`${r},${n} ${e} ${s},${n}`} />
    <polyline class="dashboard-chart__line" fill="none" points=${e} />
  </g>`}function gl(t){let e=t.values.length,r=(J-E*2)/e,s=r>3?Math.min(1,r*.2):0,n=Math.max(r-s,.5),o=j-E;return x`<g class="dashboard-chart__bars">
    ${t.values.map((a,i)=>{let l=ce(a,t.min,t.max);return x`<rect x=${E+i*r+s/2} y=${l} width=${n} height=${Math.max(o-l,0)} />`})}
  </g>`}function fl(t,e){let r=t.values.length?t.values[t.values.length-1]:0,s=S(e.min)??Math.min(t.min,0),n=(S(e.max)??Math.max(t.max,r))-s,o=n>0?Math.min(Math.max((r-s)/n,0),1):0,a=J/2,i=j-E,l=Math.min(J/2,j)-E,u=w=>{let v=Math.PI-w*Math.PI;return{x:a+l*Math.cos(v),y:i-l*Math.sin(v)}},b=u(0),h=u(1),m=u(o);return x`<g class="dashboard-chart__gauge">
    <path class="dashboard-chart__gauge-track" fill="none" d=${`M ${b.x} ${b.y} A ${l} ${l} 0 0 1 ${h.x} ${h.y}`} />
    <path class="dashboard-chart__gauge-fill" fill="none" d=${`M ${b.x} ${b.y} A ${l} ${l} 0 0 1 ${m.x} ${m.y}`} />
    <line class="dashboard-chart__gauge-needle" x1=${a} y1=${i} x2=${m.x} y2=${m.y} />
  </g>`}function ml(t){let e=t.values.length,r=Hs(t.values);return e<2?x`<g class="dashboard-chart__spark dashboard-chart__spark--${r}">
      <circle class="dashboard-chart__spark-dot" cx=${Dt(0,e)} cy=${ce(t.values[0]??0,t.min,t.max)} r="1.5" />
    </g>`:x`<g class="dashboard-chart__spark dashboard-chart__spark--${r}">
    <polyline class="dashboard-chart__line" fill="none" points=${He(t.values,t.min,t.max)} />
  </g>`}function Vs(t){return t==="line"||t==="area"||t==="bar"}function yl(){return x`<g class="dashboard-chart__grid">
    ${[E,j/2,j-E].map(t=>x`<line x1=${E} y1=${t} x2=${J-E} y2=${t} />`)}
  </g>`}function wl(t){let e=t.values.length;if(t.type==="bar"){let r=(J-E*2)/e;return x`<g class="dashboard-chart__tips">
      ${t.values.map((s,n)=>x`<rect class="dashboard-chart__tip" x=${E+n*r} y=${E} width=${r} height=${j-E*2}><title>${yt(s)}</title></rect>`)}
    </g>`}if(t.type==="gauge"){let r=e?t.values[e-1]:0;return x`<g class="dashboard-chart__tips">
      <rect class="dashboard-chart__tip" x=${E} y=${E} width=${J-E*2} height=${j-E*2}><title>${yt(r)}</title></rect>
    </g>`}return x`<g class="dashboard-chart__tips">
    ${t.values.map((r,s)=>x`<circle class="dashboard-chart__tip" cx=${Dt(s,e)} cy=${ce(r,t.min,t.max)} r="2.5"><title>${yt(r)}</title></circle>`)}
  </g>`}function vl(t,e){switch(t.type){case"bar":return gl(t);case"area":return pl(t);case"gauge":return fl(t,e);case"sparkline":return ml(t);default:return hl(t)}}function _l(t,e){let r=vl(t,e);return!t.detail||t.type==="sparkline"?r:x`<g>
    ${Vs(t.type)?yl():p}
    ${r}
    ${wl(t)}
  </g>`}function xl(t,e){let r=li(t,e);if(r.values.length===0)return c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.chart.empty")}
    </div>`;let s=R(t),n=r.detail&&r.type!=="sparkline",o=n&&Vs(r.type),a=r.type==="sparkline"&&r.label,i=n?" dashboard-chart--detail":"";return c`
    <div class="dashboard-chart dashboard-chart--${r.type}${i}">
      <svg
        class="dashboard-chart__svg"
        viewBox="0 0 ${J} ${j}"
        preserveAspectRatio="none"
        role="img"
        aria-label=${t.title??d("dashboard.widget.chart.label")}
        data-test-id="dashboard-chart"
      >
        ${_l(r,s)}
      </svg>
      ${o?c`<span class="dashboard-chart__axis dashboard-chart__axis--max"
                >${yt(r.max)}</span
              ><span class="dashboard-chart__axis dashboard-chart__axis--min"
                >${yt(r.min)}</span
              >`:p}
      ${a?c`<span
              class="dashboard-chart__spark-value dashboard-chart__spark-value--${Hs(r.values)}"
              >${yt(r.values[r.values.length-1]??0)}</span
            >`:p}
    </div>
  `}var ke=class extends Us{constructor(t){if(super(t),this.it=p,t.type!==Ws.CHILD)throw Error(this.constructor.directiveName+"() can only be used in child bindings")}render(t){if(t===p||t==null)return this._t=void 0,this.it=t;if(t===ot)return t;if(typeof t!="string")throw Error(this.constructor.directiveName+"() called with a non-string value");if(t===this.it)return this._t;this.it=t;let e=[t];return e.raw=e,this._t={_$litType$:this.constructor.resultType,strings:e,values:[]}}};ke.directiveName="unsafeHTML",ke.resultType=1;var qs=je(ke);function It(t){return t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;")}function $l(t){return/^https?:\/\//i.test(t.trim())}function jt(t){let e=t;return e=e.replace(/`([^`]+)`/g,(r,s)=>`<code>${s}</code>`),e=e.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,(r,s,n)=>$l(n)?`<a href="${n}" rel="noopener noreferrer">${s}</a>`:r),e=e.replace(/\*\*([^*]+)\*\*/g,(r,s)=>`<strong>${s}</strong>`),e=e.replace(/(^|[^*])\*([^*]+)\*/g,(r,s,n)=>`${s}<em>${n}</em>`),e=e.replace(/(^|[^_])_([^_]+)_/g,(r,s,n)=>`${s}<em>${n}</em>`),e}function Al(t){let e=t.split(`
`),r=/^(#{1,6})\s+(.*)$/.exec(e[0]??"");if(r&&e.length===1){let s=r[1].length;return`<h${s}>${jt(It(r[2]))}</h${s}>`}return e.every(s=>s.startsWith(">"))?`<blockquote>${e.map(s=>jt(It(s.replace(/^>\s?/,"")))).join("<br>")}</blockquote>`:`<p>${e.map(s=>jt(It(s))).join("<br>")}</p>`}function jr(t,e){let r=t.split(`
`).map(s=>s.replace(e?/^\s*\d+\.\s+/:/^\s*[-*]\s+/,"")).map(s=>`<li>${jt(It(s))}</li>`).join("");return e?`<ol>${r}</ol>`:`<ul>${r}</ul>`}function kl(t){return t.split(`
`).every(e=>/^\s*[-*]\s+/.test(e))}function El(t){return t.split(`
`).every(e=>/^\s*\d+\.\s+/.test(e))}function Tl(t){return`<pre><code>${It(t.join(`
`))}</code></pre>`}function Gs(t){let e=t.replace(/\r\n?/g,`
`).split(`
`),r=[],s=[],n=()=>{if(s.length===0)return;let o=s.join(`
`);kl(o)?r.push(jr(o,!1)):El(o)?r.push(jr(o,!0)):r.push(Al(o)),s=[]};for(let o=0;o<e.length;o+=1){let a=e[o];if(a.startsWith("```")){n();let i=[];for(o+=1;o<e.length&&!e[o].startsWith("```");)i.push(e[o]),o+=1;r.push(Tl(i));continue}if(a.trim()===""){n();continue}s.push(a)}return n(),r.join(`
`)}function Ks(t){return t.status==="ok"?"ok":t.status==="error"?"error":"pending"}function Sl(t){return{turn:{turnId:t,items:[],status:"streaming"},textById:new Map,callById:new Map}}function Rl(t){return t.items[t.items.length-1]}function $t(t){for(let e of t.items)e.kind==="error"&&e.retryable&&!e.superseded&&(e.superseded=!0)}function G(t,e,r){let s=t.get(r);return s||(s=Sl(r),t.set(r,s),e.push(r)),s}function Fr(t,e){let r=Rl(t);if(r&&r.kind==="tools"){r.calls.push(e);return}t.items.push({kind:"tools",calls:[e]})}function Il(t){let e=new Map,r=[];for(let s of t)switch(s.type){case"turn-start":G(e,r,s.turnId);break;case"text-start":{let{turn:n,textById:o}=G(e,r,s.turnId);if($t(n),!o.has(s.id)){let a={kind:"text",id:s.id,text:"",closed:!1};o.set(s.id,a),n.items.push(a)}break}case"text-delta":{let{turn:n,textById:o}=G(e,r,s.turnId);$t(n);let a=o.get(s.id);a||(a={kind:"text",id:s.id,text:"",closed:!1},o.set(s.id,a),n.items.push(a)),a.text+=s.delta;break}case"text-end":{let n=e.get(s.turnId)?.textById.get(s.id);n&&(n.closed=!0);break}case"tool-call-start":{let{turn:n,callById:o}=G(e,r,s.turnId);if($t(n),!o.has(s.callId)){let a={callId:s.callId,name:s.name,argsText:"",status:"building"};o.set(s.callId,a),Fr(n,a)}break}case"tool-call-delta":{let n=e.get(s.turnId)?.callById.get(s.callId);n&&(n.argsText+=s.argsTextDelta);break}case"tool-call-ready":{let{turn:n,callById:o}=G(e,r,s.turnId);$t(n);let a=o.get(s.callId);a||(a={callId:s.callId,name:s.name,argsText:"",status:"building"},o.set(s.callId,a),Fr(n,a)),a.name=s.name,a.args=s.args,a.status="ready";break}case"tool-result":{let n=e.get(s.turnId)?.callById.get(s.callId);n&&(n.ok=s.ok,n.status=s.ok?"ok":"error",s.result!==void 0&&(n.result=s.result),s.error!==void 0&&(n.error=s.error));break}case"usage":{let{turn:n}=G(e,r,s.turnId);n.usage={inputTokens:s.inputTokens,outputTokens:s.outputTokens};break}case"abort":{let{turn:n}=G(e,r,s.turnId);n.status="aborted";break}case"turn-end":{let{turn:n}=G(e,r,s.turnId);if(n.stopReason!==void 0)break;n.stopReason=s.stopReason,n.status=s.stopReason==="aborted"?"aborted":"complete";break}case"error":{let{turn:n}=G(e,r,s.turnId??r[r.length-1]??"");$t(n),n.items.push({kind:"error",code:s.code,message:s.message,retryable:s.retryable,superseded:!1});break}default:break}return r.map(s=>e.get(s).turn)}function Js(t){return typeof t=="object"&&t!==null}function At(t){return typeof t=="string"?t:""}function Hr(t,e){let r=Js(e)?e:{};switch(t.startsWith("dashboard.")?t.slice(10):t){case"tab.create":{let s=At(r.title)||At(r.slug);return s?d("dashboard.widget.chat.tool.createdTab",{name:s}):t}case"widget.add":{let s=At(r.id)||At(r.widgetId);return s?d("dashboard.widget.chat.tool.addedWidget",{id:s}):t}case"workspace.get":return d("dashboard.widget.chat.tool.readBoard");default:return t}}function Nl(t){return t.map(e=>{let r=Ks(e);return r==="ok"?"\u2713":r==="error"?"\u2717":"\xB7"}).join("")}function Ml(t){return t===1?d("dashboard.widget.chat.actionsOne"):d("dashboard.widget.chat.actionsMany",{count:String(t)})}function Vr(t){try{return JSON.stringify(t,null,2)}catch{return String(t)}}function Cl(t,e){let r=(t.status==="building"||t.status==="ready")&&!t.ok;if(r&&!e)return c`<div class="dashboard-chat__tool-row dashboard-chat__tool-row--building">
      <span class="dashboard-chat__shimmer"></span>
      <span class="dashboard-chat__tool-name">${Hr(t.name,t.args)}</span>
      <span class="dashboard-chat__tool-note">${d("dashboard.widget.chat.building")}</span>
    </div>`;let s=Ks(t),n=t.args!==void 0||t.argsText.length>0,o=t.result!==void 0||t.error!==void 0;return c`<div
    class="dashboard-chat__tool-row"
    data-status=${e&&r?"cancelled":s}
  >
    <span class="dashboard-chat__tool-name">
      <span class="dashboard-chat__tool-mark" aria-hidden="true"
        >${s==="ok"?"\u2713":s==="error"?"\u2717":"\xB7"}</span
      >
      ${Hr(t.name,t.args)}
    </span>
    ${n?c`<details class="dashboard-chat__tool-detail">
            <summary>${d("dashboard.widget.chat.args")}</summary>
            <pre>${t.args!==void 0?Vr(t.args):t.argsText}</pre>
          </details>`:p}
    ${o?c`<details class="dashboard-chat__tool-detail">
            <summary>${d("dashboard.widget.chat.result")}</summary>
            <pre>${Vr(t.error??t.result)}</pre>
          </details>`:p}
  </div>`}function Ol(t,e){let r=t.calls.length;return c`<details class="dashboard-chat__tools" data-test-id="dashboard-chat-tools">
    <summary class="dashboard-chat__chip">
      <span aria-hidden="true">🔧</span>
      <span class="dashboard-chat__chip-count">${Ml(r)}</span>
      <span class="dashboard-chat__chip-sep" aria-hidden="true">·</span>
      <span class="dashboard-chat__chip-marks">${Nl(t.calls)}</span>
    </summary>
    <div class="dashboard-chat__tool-log">
      ${t.calls.map(s=>Cl(s,e))}
    </div>
  </details>`}function Bl(t){let e=t.status==="aborted";return c`<div
    class="dashboard-chat__turn dashboard-chat__turn--assistant"
    data-test-id="dashboard-chat-turn"
    data-status=${t.status}
  >
    <div class="dashboard-chat__role">${d("dashboard.widget.chat.roleAssistant")}</div>
    ${t.items.map(r=>r.kind==="text"?c`<div class="dashboard-chat__text markdown-body">
          ${qs(Gs(r.text))}
        </div>`:r.kind==="tools"?Ol(r,e):c`<div
        class="dashboard-chat__error"
        role="alert"
        data-test-id="dashboard-chat-error"
      >
        <span class="dashboard-chat__error-message">${r.message}</span>
        ${r.retryable&&r.superseded?c`<span class="dashboard-chat__error-retry"
                >${d("dashboard.widget.chat.retrying")}</span
              >`:p}
      </div>`)}
  </div>`}function qr(t){return c`<div
    class="dashboard-chat__turn dashboard-chat__turn--user"
    data-test-id="dashboard-chat-user"
  >
    <div class="dashboard-chat__role">${d("dashboard.widget.chat.roleUser")}</div>
    <div class="dashboard-chat__text">${t}</div>
  </div>`}var Dl=100,Ll=class{constructor(t){this.widgetId=t,this.root=null,this.ctx=null,this.widget=null,this.events=[],this.unsubscribe=null,this.userMessages=new Map,this.pendingUserText=null,this.sending=!1,this.stickToBottom=!0,this.rootRef=e=>{e instanceof HTMLElement?this.mount(e):this.destroy()},this.onSubmit=e=>{e.preventDefault(),this.send()},this.onTextareaKey=e=>{e.key==="Enter"&&!e.shiftKey&&(e.preventDefault(),this.send())},this.onStop=e=>{this.ctx?.chat?.abort(e).catch(()=>{})},this.onScroll=e=>{let r=e.currentTarget;this.stickToBottom=r.scrollHeight-r.scrollTop-r.clientHeight<Dl,this.updateJumpPill()},this.jumpToLatest=()=>{let e=this.root?.querySelector(".dashboard-chat__scroll");e&&(this.stickToBottom=!0,e.scrollTop=e.scrollHeight,this.updateJumpPill())}}setContext(t,e){this.ctx=t,this.widget=e,this.root&&this.renderIsland()}mount(t){this.root=t,this.unsubscribe?.(),this.unsubscribe=null,this.events=[],this.userMessages.clear(),this.pendingUserText=null,this.sending=!1,this.stickToBottom=!0,this.renderIsland();let e=this.ctx?.chat;e&&(e.history().then(r=>{this.events=[...r,...this.events],this.renderIsland()}).catch(()=>{}),this.unsubscribe=e.subscribe(r=>{this.events.push(r),this.renderIsland()}))}destroy(){this.unsubscribe?.(),this.unsubscribe=null,this.root=null,Ee.delete(this.widgetId)}liveTurnId(t){for(let e=t.length-1;e>=0;e-=1)if(t[e].status==="streaming")return t[e].turnId}send(){let t=this.ctx?.chat,e=this.root?.querySelector(".dashboard-chat__textarea");if(!t||!e)return;let r=e.value.trim();!r||this.sending||(e.value="",this.pendingUserText=r,this.sending=!0,this.stickToBottom=!0,this.renderIsland(),t.send(r).then(({turnId:s})=>{this.userMessages.set(s,r)}).catch(()=>{}).finally(()=>{this.pendingUserText=null,this.sending=!1,this.renderIsland()}))}updateJumpPill(){let t=this.root?.querySelector(".dashboard-chat__jump");t&&(t.hidden=this.stickToBottom)}renderIsland(){if(!this.root)return;let t=Il(this.events),e=this.liveTurnId(t),r=e!==void 0||this.sending,s=this.ctx?.registryPending??[],n=!!this.ctx?.approveWidget,o=r&&n&&s.length>0,a=t.length===0&&this.pendingUserText===null,i=!this.ctx?.chat;if(Ce(c`
        <div class="dashboard-chat__scroll" @scroll=${this.onScroll}>
          ${a?c`<div class="dashboard-chat__empty" data-test-id="dashboard-chat-empty">
                  ${d("dashboard.widget.chat.empty")}
                </div>`:p}
          ${t.map(l=>{let u=this.userMessages.get(l.turnId);return c`${u!==void 0?qr(u):p}
            ${Bl(l)}`})}
          ${this.pendingUserText!==null?qr(this.pendingUserText):p}
          ${o?s.map(l=>c`<div
                      class="dashboard-chat__approval"
                      data-test-id="dashboard-chat-approval"
                    >
                      <span class="dashboard-chat__approval-title"
                        >${d("dashboard.widget.chat.approveTitle",{name:l})}</span
                      >
                      <span class="dashboard-chat__approval-actions">
                        <button
                          class="bs-btn bs-btn--small bs-btn--primary"
                          type="button"
                          data-test-id="dashboard-chat-approve"
                          @click=${()=>this.ctx?.approveWidget?.(l,"approved")}
                        >
                          ${d("dashboard.widget.chat.approve")}
                        </button>
                        <button
                          class="bs-btn bs-btn--small"
                          type="button"
                          data-test-id="dashboard-chat-reject"
                          @click=${()=>this.ctx?.approveWidget?.(l,"rejected")}
                        >
                          ${d("dashboard.widget.chat.reject")}
                        </button>
                      </span>
                    </div>`):p}
        </div>
        <button
          class="dashboard-chat__jump"
          type="button"
          hidden
          data-test-id="dashboard-chat-jump"
          @click=${this.jumpToLatest}
        >
          ${d("dashboard.widget.chat.jumpToLatest")} ↓
        </button>
        <form class="dashboard-chat__input" @submit=${this.onSubmit}>
          <textarea
            class="dashboard-chat__textarea"
            data-test-id="dashboard-chat-textarea"
            rows="2"
            ?disabled=${i}
            placeholder=${this.placeholder()}
            @keydown=${this.onTextareaKey}
          ></textarea>
          <div class="dashboard-chat__input-actions">
            ${e!==void 0?c`<button
                    class="bs-btn bs-btn--small dashboard-chat__stop"
                    type="button"
                    data-test-id="dashboard-chat-stop"
                    @click=${()=>this.onStop(e)}
                  >
                    ${d("dashboard.widget.chat.stop")}
                  </button>`:p}
            <button
              class="bs-btn bs-btn--small bs-btn--primary dashboard-chat__send"
              type="submit"
              data-test-id="dashboard-chat-send"
              ?disabled=${i}
            >
              ${d("dashboard.widget.chat.send")}
            </button>
          </div>
        </form>
        ${i?c`<div class="dashboard-chat__hint" data-test-id="dashboard-chat-disconnected">
                ${d("dashboard.widget.chat.disconnected")}
              </div>`:p}
      `,this.root),this.stickToBottom){let l=this.root.querySelector(".dashboard-chat__scroll");l&&(l.scrollTop=l.scrollHeight)}this.updateJumpPill()}placeholder(){return At((Js(this.widget?.props)?this.widget.props:{}).placeholder)||d("dashboard.widget.chat.placeholder")}},Ee=new Map;function Pl(t,e,r){let s=Ee.get(t.id);return s||(s=new Ll(t.id),Ee.set(t.id,s)),s.setContext(r,t),c`<div
    class="dashboard-chat"
    data-test-id="dashboard-chat"
    ${Bt(s.rootRef)}
  ></div>`}function Wl(t){return t==="ok"?"dashboard-badge--ok":t==="error"?"dashboard-badge--error":"dashboard-badge--muted"}function Ul(t,e){let r=Ja(t,e);return r.jobs.length===0?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.cron.empty")}
    </div>`:c`
    <ul class="dashboard-list dashboard-cron" data-test-id="dashboard-cron">
      ${r.jobs.map(s=>c`
          <li class="dashboard-list__row ${s.enabled?"":"dashboard-list__row--disabled"}">
            <span class="dashboard-list__label">${s.name}</span>
            <span class="dashboard-list__meta">
              ${s.nextRunAtMs!==null?d("dashboard.widget.cron.next",{time:Fe(s.nextRunAtMs)}):d("dashboard.widget.cron.noNext")}
            </span>
            ${s.lastStatus?c`<span class="dashboard-badge ${Wl(s.lastStatus)}"
                    >${s.lastStatus}</span
                  >`:p}
          </li>
        `)}
    </ul>
  `}function Xs(t){return t==="scripts"?"allow-scripts":""}function zl(t,e,r){let s=Rs(R(t).url,{allowExternalEmbedUrls:r.embed.allowExternalEmbedUrls});return s.status==="missing"?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.embed.missing")}
    </div>`:s.status==="blocked"?c`<div class="dashboard-widget__placeholder" data-test-id="dashboard-embed-blocked">
      ${s.reason==="external"?d("dashboard.widget.embed.blockedExternal"):d("dashboard.widget.embed.blockedScheme")}
    </div>`:c`<iframe
    class="dashboard-embed__frame"
    data-test-id="dashboard-embed-frame"
    src=${s.url}
    title=${t.title}
    sandbox=${Xs(r.embed.embedSandboxMode)}
    referrerpolicy="no-referrer"
    loading="lazy"
  ></iframe>`}function jl(t,e){let r=ti(t,e);return r.instances.length===0?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.instances.empty")}
    </div>`:c`
    <ul class="dashboard-list dashboard-instances" data-test-id="dashboard-instances">
      ${r.instances.map(s=>c`
          <li class="dashboard-list__row">
            <span
              class="dashboard-dot ${s.healthy?"dashboard-dot--ok":"dashboard-dot--warn"}"
              aria-hidden="true"
            ></span>
            <span class="dashboard-list__label">${s.id}</span>
            ${s.detail?c`<span class="dashboard-list__meta">${s.detail}</span>`:p}
            ${s.lastInputMs!==null?c`<span class="dashboard-list__meta"
                    >${d("dashboard.widget.instances.idle",{duration:el(s.lastInputMs)})}</span
                  >`:p}
          </li>
        `)}
    </ul>
  `}function Fl(t,e){let r=Da(t,e);return r.trim()?c`<div class="dashboard-markdown markdown-body">
    ${qs(Gs(r))}
  </div>`:c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.markdownEmpty")}
    </div>`}function Hl(t){let e=R(t);return typeof e.text=="string"?e.text:""}function Vl(t){return e=>{if(!(e instanceof HTMLTextAreaElement))return;let r=e;if(r.dataset.notesBound==="1")return;r.dataset.notesBound="1",t.get().then(n=>{r.dataset.notesDirty!=="1"&&(r.value=ci(n.state))}).catch(()=>{});let s;r.addEventListener("input",()=>{r.dataset.notesDirty="1";let n=r.value;s!==void 0&&clearTimeout(s),s=setTimeout(()=>{t.set(n).catch(()=>{})},500)})}}function ql(t,e,r){let s=d("dashboard.widget.notes.placeholder");if(!r.state){let n=Hl(t);return c`
      <div class="dashboard-notes dashboard-notes--readonly" data-test-id="dashboard-notes">
        <textarea
          class="dashboard-notes__pad"
          data-test-id="dashboard-notes-pad"
          readonly
          aria-label=${t.title}
          placeholder=${s}
        >
${n}</textarea>
        <div class="dashboard-notes__hint" data-test-id="dashboard-notes-hint">
          ${d("dashboard.widget.notes.readonlyHint")}
        </div>
      </div>
    `}return c`
    <div class="dashboard-notes" data-test-id="dashboard-notes">
      <textarea
        class="dashboard-notes__pad"
        data-test-id="dashboard-notes-pad"
        aria-label=${t.title}
        placeholder=${s}
        ${Bt(Vl(r.state))}
      ></textarea>
    </div>
  `}var Gl=["desktop","tablet","mobile"];function Gr(t){return`dashboard-preview__frame-wrap dashboard-preview__frame-wrap--${t}`}function Kl(t,e,r){let s=Rs(R(t).url,{allowExternalEmbedUrls:r.embed.allowExternalEmbedUrls});if(s.status==="missing")return c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.preview.missing")}
    </div>`;if(s.status==="blocked")return c`<div class="dashboard-widget__placeholder" data-test-id="dashboard-preview-blocked">
      ${s.reason==="external"?d("dashboard.widget.preview.blockedExternal"):d("dashboard.widget.preview.blockedScheme")}
    </div>`;let n=wi(t),o=Ur(),a=Ur(),i=()=>{let u=o.value;if(u){let b=u.getAttribute("src");b!==null&&u.setAttribute("src",b)}},l=u=>{let b=a.value;b&&(b.className=Gr(u))};return c`<div class="dashboard-preview">
    <div class="dashboard-preview__toolbar" role="toolbar">
      <div class="dashboard-preview__viewports" role="group">
        ${Gl.map(u=>c`<button
              class="dashboard-preview__viewport"
              type="button"
              data-test-id=${`dashboard-preview-viewport-${u}`}
              data-viewport=${u}
              title=${d(`dashboard.widget.preview.viewport.${u}`)}
              aria-label=${d(`dashboard.widget.preview.viewport.${u}`)}
              @click=${()=>l(u)}
            >
              ${d(`dashboard.widget.preview.viewport.${u}`)}
            </button>`)}
      </div>
      <button
        class="dashboard-preview__reload"
        type="button"
        data-test-id="dashboard-preview-reload"
        title=${d("dashboard.widget.preview.reload")}
        aria-label=${d("dashboard.widget.preview.reload")}
        @click=${i}
      >
        ${d("dashboard.widget.preview.reload")}
      </button>
    </div>
    <div class=${Gr(n)} ${Bt(a)}>
      <iframe
        class="dashboard-embed__frame dashboard-preview__frame"
        data-test-id="dashboard-preview-frame"
        ${Bt(o)}
        src=${s.url}
        title=${t.title}
        sandbox=${Xs(r.embed.embedSandboxMode)}
        referrerpolicy="no-referrer"
        loading="lazy"
      ></iframe>
    </div>
  </div>`}function Jl(t,e,r){let s=Va(t,e);if(s.rows.length===0)return c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.sessions.empty")}
    </div>`;let n=a=>r?.sessionHref?.(a)??"#",o=r?.onNavigate;return c`
    <ul class="dashboard-list dashboard-sessions" data-test-id="dashboard-sessions">
      ${s.rows.map(a=>c`
          <li class="dashboard-list__row">
            <a
              class="dashboard-list__link"
              href=${n(a.key)}
              @click=${o?i=>{i.preventDefault(),o(a.key)}:p}
            >
              <span
                class="dashboard-dot ${a.active?"dashboard-dot--live":""}"
                aria-hidden="true"
              ></span>
              <span class="dashboard-list__label">${a.label}</span>
              ${a.updatedAt!==null?c`<span class="dashboard-list__meta"
                      >${Fe(a.updatedAt)}</span
                    >`:p}
            </a>
          </li>
        `)}
    </ul>
  `}function Xl(t,e){let r=Ba(t,e);return c`
    <div class="dashboard-stat">
      <div class="dashboard-stat__value">${r.display??d("dashboard.widget.stat.empty")}</div>
      ${r.label?c`<div class="dashboard-stat__label">${r.label}</div>`:p}
    </div>
  `}function Yl(t){return t==null?"":typeof t=="string"?t:typeof t=="number"||typeof t=="boolean"?String(t):JSON.stringify(t)}function Zl(t,e){let r=za(t,e);if(r.total===0||r.columns.length===0)return c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.table.empty")}
    </div>`;let s=r.total-r.shown;return c`
    <div class="dashboard-table">
      <table class="dashboard-table__grid">
        <thead>
          <tr>
            ${r.columns.map(n=>c`<th scope="col">${n}</th>`)}
          </tr>
        </thead>
        <tbody>
          ${r.rows.map(n=>c`
              <tr>
                ${r.columns.map(o=>c`<td>${Yl(n[o])}</td>`)}
              </tr>
            `)}
        </tbody>
      </table>
      ${s>0?c`<div class="dashboard-table__footer">
              ${d("dashboard.widget.table.more",{count:String(s)})}
            </div>`:p}
    </div>
  `}function Ql(t,e){let r=qa(t,e);return c`
    <div class="dashboard-usage" data-test-id="dashboard-usage">
      <div class="dashboard-usage__metric">
        <div class="dashboard-usage__value">${Qd(r.cost)}</div>
        <div class="dashboard-usage__label">${d("dashboard.widget.usage.cost")}</div>
      </div>
      <div class="dashboard-usage__metric">
        <div class="dashboard-usage__value">${tl(r.tokens)}</div>
        <div class="dashboard-usage__label">${d("dashboard.widget.usage.tokens")}</div>
      </div>
    </div>
  `}var tc={"stat-card":(t,e)=>Xl(t,e),markdown:(t,e)=>Fl(t,e),table:(t,e)=>Zl(t,e),"iframe-embed":zl,preview:Kl,sessions:(t,e,r)=>Jl(t,e,r),usage:(t,e)=>Ql(t,e),cron:(t,e)=>Ul(t,e),instances:(t,e)=>jl(t,e),activity:(t,e)=>sl(t,e),chart:(t,e)=>xl(t,e),notes:ql,"action-form":Zd,"action-button":Jd,"agent-status":(t,e)=>nl(t,e),approvals:ul,chat:Pl};function ec(t){let e=t.startsWith("builtin:")?t.slice(8):t;return tc[e]}function rc(t){return t.replace(/\s*\(custom\)\s*$/iu,"").trim()||t}function sc(t,e){let r=X(t.createdBy);return r?e?c`<span
      class=${e.dimmed?"dashboard-widget__agent dashboard-widget__agent--dimmed":"dashboard-widget__agent"}
      style="--dashboard-agent-hue: ${e.hue}"
      data-test-id="dashboard-widget-agent-chip"
      data-agent=${e.actor}
      title=${d("dashboard.widget.agentChipTooltip",{agent:e.actor})}
      >${e.short}</span
    >`:c`<span
    class="dashboard-widget__provenance"
    title=${d("dashboard.widget.provenanceTooltip",{agent:r})}
    >${d("dashboard.widget.provenanceChip")}</span
  >`:p}function nc(t){return t.ephemeral?c`<span
    class="dashboard-widget__ephemeral"
    data-test-id="dashboard-widget-ephemeral"
    title=${d("dashboard.widget.ephemeralTooltip")}
    >${d("dashboard.widget.ephemeralBadge")}</span
  >`:p}function oc(t){return c`
    <div class="dashboard-widget__blame" role="note" data-test-id="dashboard-widget-blame">
      <span class="dashboard-widget__blame-text">${t.firstSeenVersion!==void 0?d("dashboard.widget.blame.createdByVersion",{actor:t.actor,version:String(t.firstSeenVersion)}):d("dashboard.widget.blame.createdBy",{actor:t.actor})}</span>
      ${t.agentId!==null&&t.logbookHref?c`<a
              class="dashboard-widget__blame-link"
              href=${t.logbookHref}
              target="_blank"
              rel="noopener noreferrer"
              data-test-id="dashboard-widget-blame-link"
              >${M.externalLink} ${d("dashboard.widget.blame.logbookLink")}</a
            >`:p}
    </div>
  `}function ac(t,e,r){return c`
    <div class="dashboard-widget__menu" role="menu">
      ${r?oc(r):p}
      ${t.ephemeral?c`<button
              class="dashboard-widget__menu-item"
              type="button"
              role="menuitem"
              data-test-id="dashboard-widget-pin"
              @click=${()=>e.onPin(t)}
            >
              ${d("dashboard.widget.menu.pin")}
            </button>`:p}
      <button
        class="dashboard-widget__menu-item"
        type="button"
        role="menuitem"
        @click=${()=>e.onEditTitle(t)}
      >
        ${d("dashboard.widget.menu.editTitle")}
      </button>
      <button
        class="dashboard-widget__menu-item"
        type="button"
        role="menuitem"
        @click=${()=>e.onMoveToTab(t)}
      >
        ${d("dashboard.widget.menu.moveToTab")}
      </button>
      <button
        class="dashboard-widget__menu-item"
        type="button"
        role="menuitem"
        @click=${()=>e.onHide(t)}
      >
        ${d("dashboard.widget.menu.hide")}
      </button>
      <button
        class="dashboard-widget__menu-item dashboard-widget__menu-item--danger"
        type="button"
        role="menuitem"
        @click=${()=>e.onRemove(t)}
      >
        ${d("dashboard.widget.menu.remove")}
      </button>
    </div>
  `}function ic(t,e,r){if(e&&"error"in e)throw new Error(e.error);let s=e&&"value"in e?e.value:void 0,n=ec(t.kind);return n?n(t,s,r):t.kind.startsWith("custom:")?c`<div class="dashboard-widget__placeholder">
      ${d("dashboard.widget.customPlaceholder")}
    </div>`:c`<div class="dashboard-widget__placeholder">
    ${d("dashboard.widget.unknownKind",{kind:t.kind})}
  </div>`}function dc(t,e){if(e.status==="approved")return e.manifest?Fd({widget:t,manifest:e.manifest,context:e.host}):c`<div
        class="dashboard-widget__placeholder"
        data-test-id="dashboard-custom-loading"
      >
        ${d("dashboard.widget.customLoading")}
      </div>`;if(e.status==="pending"){let r=X(t.createdBy);return c`
      <div
        class="dashboard-widget__approval"
        role="group"
        data-test-id="dashboard-custom-pending"
        aria-label=${d("dashboard.widget.approval.title")}
      >
        <div class="dashboard-widget__approval-title">${d("dashboard.widget.approval.title")}</div>
        <div class="dashboard-widget__approval-sub">
          ${r?d("dashboard.widget.approval.byAgent",{agent:r}):d("dashboard.widget.approval.byUnknown")}
        </div>
        <div class="dashboard-widget__approval-actions">
          <button
            class="bs-btn bs-btn--small bs-btn--primary"
            type="button"
            data-test-id="dashboard-custom-approve"
            @click=${()=>e.onApprove(t)}
          >
            ${d("dashboard.widget.approval.approve")}
          </button>
          <button
            class="bs-btn bs-btn--small"
            type="button"
            data-test-id="dashboard-custom-reject"
            @click=${()=>e.onReject(t)}
          >
            ${d("dashboard.widget.approval.reject")}
          </button>
        </div>
      </div>
    `}return c`<div class="dashboard-widget__placeholder" data-test-id="dashboard-custom-rejected">
    ${d("dashboard.widget.approval.unavailable")}
  </div>`}function Ys(t,e,r,s,n){try{return t.kind.startsWith("custom:")&&n?dc(t,n):ic(t,e,r)}catch(o){let a=o instanceof Error?o.message:String(o);return c`
      <div class="dashboard-widget__error" role="alert" data-test-id="dashboard-widget-error">
        <div class="dashboard-widget__error-title">${d("dashboard.widget.errorTitle")}</div>
        <div class="dashboard-widget__error-humane">${d("dashboard.widget.errorHumane")}</div>
        <details class="dashboard-widget__error-detail">
          <summary>${d("dashboard.widget.errorDetailSummary")}</summary>
          <div class="dashboard-widget__error-message">${a}</div>
        </details>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          @click=${()=>s.onRemove(t)}
        >
          ${d("dashboard.widget.menu.remove")}
        </button>
      </div>
    `}}function lc(t){let{widget:e,callbacks:r}=t,s=["dashboard-widget",e.collapsed?"dashboard-widget--collapsed":"",t.pending?"dashboard-widget--pending":"",t.dragging?"dashboard-widget--dragging":"",t.dragging&&t.dragTransform?"dashboard-widget--carried":"",t.agentChip?.dimmed?"dashboard-widget--agent-dimmed":""].filter(Boolean).join(" "),n=Ue(e.grid);return c`
    <section
      class=${s}
      style=${t.dragging&&t.dragTransform?`${n}; transform: ${t.dragTransform}`:n}
      data-widget-id=${e.id}
      data-test-id="dashboard-widget"
    >
      <header
        class="dashboard-widget__bar"
        @pointerdown=${o=>r.onMovePointerDown(e,o)}
      >
        <button
          class="dashboard-widget__collapse"
          type="button"
          aria-expanded=${e.collapsed?"false":"true"}
          aria-label=${e.collapsed?d("dashboard.widget.expand"):d("dashboard.widget.collapse")}
          @pointerdown=${o=>o.stopPropagation()}
          @click=${()=>r.onToggleCollapse(e)}
        >
          ${e.collapsed?M.chevronRight:M.chevronDown}
        </button>
        <span class="dashboard-widget__title" title=${e.title}
          >${rc(e.title)}</span
        >
        ${sc(e,t.agentChip)} ${nc(e)}
        <span
          class="dashboard-widget__handle"
          role="button"
          tabindex="0"
          aria-label=${d("dashboard.widget.moveHandle")}
          @keydown=${o=>Kr(o,e,"move",r)}
          >${M.arrowUpDown}</span
        >
        <button
          class="dashboard-widget__menu-toggle"
          type="button"
          aria-haspopup="menu"
          aria-expanded=${t.menuOpen?"true":"false"}
          aria-label=${d("dashboard.widget.menuLabel")}
          @pointerdown=${o=>o.stopPropagation()}
          @click=${()=>r.onToggleMenu(e)}
        >
          ${M.moreHorizontal}
        </button>
        ${t.menuOpen?ac(e,r,t.blame):p}
      </header>
      ${e.collapsed?p:c`
              <div class="dashboard-widget__body">
                ${Ys(e,t.binding,t.builtinContext,r,t.custom)}
              </div>
              <span
                class="dashboard-widget__resize"
                role="button"
                tabindex="0"
                aria-label=${d("dashboard.widget.resizeHandle")}
                @pointerdown=${o=>r.onResizePointerDown(e,o)}
                @keydown=${o=>Kr(o,e,"resize",r)}
              ></span>
            `}
    </section>
  `}function Kr(t,e,r,s){let n=t.key==="ArrowLeft"?"left":t.key==="ArrowRight"?"right":t.key==="ArrowUp"?"up":t.key==="ArrowDown"?"down":null;n&&(t.preventDefault(),s.onKeyboardNudge(e,r,n))}var Jr=10;function Zs(t){let e=2166136261;for(let r=0;r<t.length;r++)e^=t.charCodeAt(r),e=Math.imul(e,16777619);return(e>>>0)%360}function Qs(t){return t.length<=Jr?t:`${t.slice(0,Jr-1)}\u2026`}function tn(t){let e=new Set;for(let r of t.tabs)for(let s of r.widgets){let n=s.createdBy;n&&X(n)&&e.add(n)}return[...e].sort()}function cc(t,e){let r=X(t);return r?{actor:t,agentId:r,short:Qs(r),hue:Zs(t),dimmed:e!==null&&t!==e}:null}var uc={embedSandboxMode:"strict",allowExternalEmbedUrls:!1};function bc(t){return t?{embedSandboxMode:t.sandboxMode,allowExternalEmbedUrls:t.allowExternalUrls}:uc}function hc(){return{open:!1,loading:!1,error:null,entries:[],snapshots:new Map,selectedVersion:null,confirmRestore:!1,restoring:!1}}var en="boardstate:gallery-url:v1";function pc(t){try{return t?.getItem(en)??""}catch{return""}}function gc(t,e){try{t?.setItem(en,e)}catch{}}function bt(t){return t instanceof Error&&t.message.trim()?t.message.trim():"Widget gallery error."}var rn="boardstate:onboarding-dismissed:v1";function fc(t){try{return t?.getItem(rn)==="1"}catch{return!1}}function mc(t){try{t?.setItem(rn,"1")}catch{}}var Zt=new WeakMap,Qt=new WeakMap;function Te(t){let e=Qt.get(t);e&&(document.removeEventListener("pointerdown",e.onPointerDown,!0),document.removeEventListener("keydown",e.onKeyDown,!0),Qt.delete(t))}function yc(t,e,r){let s=e.openMenuWidgetId!==null;if(s===Qt.has(t))return;if(!s){Te(t);return}let n=()=>{e.openMenuWidgetId!==null&&(e.openMenuWidgetId=null,Te(t),r())},o=i=>{let l=i.target;l instanceof Element&&l.closest(".dashboard-widget__menu, .dashboard-widget__menu-toggle")||n()},a=i=>{i.key==="Escape"&&(i.preventDefault(),n())};document.addEventListener("pointerdown",o,!0),document.addEventListener("keydown",a,!0),Qt.set(t,{onPointerDown:o,onKeyDown:a})}function wc(t){Te(t),vc(t)}function vc(t){let e=Zt.get(t);if(e){for(let r of e.streamSubs.values())r.unsubscribe();e.streamSubs.clear()}}function _c(t,e){let r=Zt.get(t);return r||(r={openMenuWidgetId:null,drag:null,bindingResults:new Map,bindingLoads:new Set,bindingVersion:-1,streamSubs:new Map,streamValues:new Map,manifestCache:new Map,manifestLoads:new Set,dataVersion:0,dialog:null,onboardingDismissed:fc(e),collapsedTabGroups:new Set,lastPresenceSlug:null,history:hc(),gallery:null,highlightedAgent:null},Zt.set(t,r)),r}function xc(t){let e=Zt.get(t);e&&(e.dataVersion+=1)}function sn(t){let e=t.bindings;return e?Object.values(e)[0]??null:null}function $c(t,e){return t.workspaceVersion*1000003+e.dataVersion}function Ac(t,e,r,s,n){if(!e){for(let a of t.streamSubs.values())a.unsubscribe();t.streamSubs.clear();return}let o=new Map;for(let a of s.widgets){let i=sn(a);i?.source==="stream"&&i.event&&o.set(a.id,i)}for(let[a,i]of t.streamSubs){let l=o.get(a);(!l||i.workspaceVersion!==r.workspaceVersion||i.event!==l.event||i.pointer!==l.pointer)&&(i.unsubscribe(),t.streamSubs.delete(a),t.streamValues.delete(a))}for(let[a,i]of o){if(t.streamSubs.has(a))continue;let l=xd(e,i,u=>{t.streamValues.set(a,u),t.bindingResults.set(a,u),n?.()});t.streamSubs.set(a,{workspaceVersion:r.workspaceVersion,event:i.event,...i.pointer!==void 0?{pointer:i.pointer}:{},unsubscribe:l})}}async function kc(t,e,r){let s=e.bindings??{},n=[];for(let o of r.inputs??[]){let a=s[o];if(!a)return{error:`Computed input not found: ${o}`};let i=await ze(t,a);if("error"in i)return{error:i.error};n.push(i.value)}return _d(r.op??"",n,r.arg)}function Ec(t,e,r,s,n){let o=$c(r,t);t.bindingVersion!==o&&(t.bindingResults.clear(),t.bindingLoads.clear(),t.bindingVersion=o),Ac(t,e,r,s,n);for(let a of s.widgets){let i=sn(a);if(!(!i||t.bindingResults.has(a.id)||t.bindingLoads.has(a.id))){if(i.source==="stream"){let l=t.streamValues.get(a.id);l&&t.bindingResults.set(a.id,l);continue}t.bindingLoads.add(a.id),(i.source==="computed"?kc(e,a,i):ze(e,i)).then(l=>{t.bindingResults.set(a.id,l),t.bindingLoads.delete(a.id),n?.()})}}}function Tc(t){return{width:(t instanceof HTMLElement?t.querySelector(".dashboard-grid"):null)?.clientWidth??0}}function Sc(t){if(t.key!=="Escape")return;let e=t.currentTarget.closest("details");e?.open&&(t.preventDefault(),e.open=!1,e.querySelector("summary")?.focus())}function Rc(t){let e=t.currentTarget;if(!e.open)return;let r=n=>{n.target instanceof Node&&e.contains(n.target)||(e.open=!1,document.removeEventListener("pointerdown",r,!0))},s=()=>{e.open||(document.removeEventListener("pointerdown",r,!0),e.removeEventListener("toggle",s))};document.addEventListener("pointerdown",r,!0),e.addEventListener("toggle",s)}function Ic(t,e,r,s){if(e.onboardingDismissed||r.tabs.some(o=>o.widgets.length>0))return p;let n=()=>{e.onboardingDismissed=!0,mc(t.storage),s()};return c`
    <div class="dashboard-onboarding" role="note" data-test-id="dashboard-onboarding">
      <span class="dashboard-onboarding__icon" aria-hidden="true">${M.spark}</span>
      <div class="dashboard-onboarding__body">
        <div class="dashboard-onboarding__title">${d("dashboard.onboarding.title")}</div>
        <div class="dashboard-onboarding__sub">${d("dashboard.onboarding.primary")}</div>
        <div class="dashboard-onboarding__sub">
          ${d("dashboard.onboarding.secondary")}
          <code class="dashboard-onboarding__cmd">${d("dashboard.empty.onboardingCommand")}</code>
        </div>
      </div>
      <button
        class="dashboard-onboarding__dismiss"
        type="button"
        data-test-id="dashboard-onboarding-dismiss"
        aria-label=${d("common.dismiss")}
        @click=${n}
      >
        ${M.x}
      </button>
    </div>
  `}function nn(t,e,r,s){e.activeSlug=_s(r,s),t.onNavigate?.(s),t.onRequestUpdate?.()}function Nc(){return c`<svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
  </svg>`}function Mc(t){if(t<=0)return p;let e=d("dashboard.tabs.presence",{count:String(t)});return c`
    <span
      class="dashboard-tab__presence"
      data-test-id="dashboard-tab-presence"
      title=${e}
      aria-label=${e}
    >
      <span class="dashboard-tab__presence-dot" aria-hidden="true"></span>
      ${t>1?c`<span class="dashboard-tab__presence-count">${t}</span>`:p}
    </span>
  `}function Xr(t,e,r,s,n,o=0){return c`
    <button
      class="dashboard-tab ${n?"dashboard-tab--active":""}"
      type="button"
      role="tab"
      aria-selected=${n?"true":"false"}
      data-test-id="dashboard-tab"
      data-ws=${s.slug}
      @click=${()=>nn(t,e,r,s.slug)}
    >
      ${s.icon&&Object.hasOwn(M,s.icon)?c`<span class="dashboard-tab__icon" aria-hidden="true"
              >${M[s.icon]}</span
            >`:p}
      <span class="dashboard-tab__label">${s.title}</span>
      ${s.visibility==="private"?c`<span
              class="dashboard-tab__private"
              data-test-id="dashboard-tab-private"
              title=${d("dashboard.tabs.private")}
              aria-label=${d("dashboard.tabs.private")}
              >${Nc()}</span
            >`:p}
      ${Mc(o)}
    </button>
  `}function Cc(t){return t.kind==="agent"?d("dashboard.tabs.groupAgent",{agent:t.agentId??"agent"}):t.kind==="system"?d("dashboard.tabs.groupSystem"):d("dashboard.tabs.groupUser")}function Oc(t,e,r,s){let n=()=>t.onRequestUpdate?.(),o=oe(s),a=ta(o),i=Qo(s),l=a.length>1,u=b=>Gi(t.host,b).length;return c`
    <nav class="dashboard-tabs" role="tablist" aria-label=${d("dashboard.tabs.label")}>
      ${l?a.map(b=>{let h=r.collapsedTabGroups.has(b.key),m=()=>{h?r.collapsedTabGroups.delete(b.key):r.collapsedTabGroups.add(b.key),n()},w=Cc(b);return c`
                <div
                  class="dashboard-tab-group ${h?"dashboard-tab-group--collapsed":""}"
                  data-test-id="dashboard-tab-group"
                  data-group=${b.key}
                >
                  <button
                    class="dashboard-tab-group__toggle"
                    type="button"
                    data-test-id="dashboard-tab-group-toggle"
                    aria-expanded=${h?"false":"true"}
                    aria-label=${h?d("dashboard.tabs.expandGroup",{group:w}):d("dashboard.tabs.collapseGroup",{group:w})}
                    @click=${m}
                  >
                    <span class="dashboard-tab-group__chevron" aria-hidden="true"
                      >${h?M.chevronRight:M.chevronDown}</span
                    >
                    <span class="dashboard-tab-group__label">${w}</span>
                    <span class="dashboard-tab-group__count">${b.tabs.length}</span>
                  </button>
                  ${h?p:b.tabs.map(v=>Xr(t,e,s,v,v.slug===e.activeSlug,u(v.slug)))}
                </div>
              `}):o.map(b=>Xr(t,e,s,b,b.slug===e.activeSlug,u(b.slug)))}
      ${i.length>0?c`
              <details
                class="dashboard-tabs__hidden"
                @toggle=${Rc}
                @keydown=${Sc}
              >
                <summary class="dashboard-tab dashboard-tab--overflow">
                  <span class="dashboard-tab__icon" aria-hidden="true">${M.eyeOff}</span>
                  <span class="dashboard-tab__label"
                    >${d("dashboard.tabs.hidden",{count:String(i.length)})}</span
                  >
                </summary>
                <div class="dashboard-tabs__hidden-menu" role="menu">
                  ${i.map(b=>c`
                      <button
                        class="dashboard-tabs__hidden-item"
                        type="button"
                        role="menuitem"
                        @click=${()=>nn(t,e,s,b.slug)}
                      >
                        ${b.title}
                      </button>
                    `)}
                </div>
              </details>
            `:p}
    </nav>
  `}function Bc(t,e,r,s){let n=e.basePath??"";for(let o of s.widgets){let a=Le(o.kind);!a||vs(r,o.kind)!=="approved"||t.manifestCache.has(a)||t.manifestLoads.has(a)||(t.manifestLoads.add(a),Td(n,a).then(i=>{t.manifestLoads.delete(a),i&&(t.manifestCache.set(a,i),e.onRequestUpdate?.())}))}}function Dc(t){let e=t.transport,r=t.sessionKey??"main";return({widgetKey:s,text:n})=>Cs({widgetKey:s,text:n,confirmPrompt:async o=>t.confirm?await t.confirm(o):typeof window<"u"?window.confirm(o):!1,sendPrompt:async o=>{if(!e)throw new Error("Not connected.");await e.request("chat.send",{sessionKey:r,message:o,deliver:!1})}})}function on(t,e,r,s){let n=t.transport,o={embed:bc(t.embed),dispatchPrompt:Dc(t),onActionError:a=>{e.actionError=a,t.onRequestUpdate?.()},approvals:Ri(r,(a,i)=>void Kt(e,n,{name:a,decision:i}),(a,i,l)=>void hd(e,n,{name:a,decision:i,...l?.tools!==void 0?{tools:l.tools}:{},...l?.autoConfirm!==void 0?{autoConfirm:l.autoConfirm}:{},...l?.expiresAt!==void 0?{expiresAt:l.expiresAt}:{}})),registryPending:Pc(r)};return n&&(o.state=Uc(n,s.id),o.chat=Wc(n,t.sessionKey??"main"),o.approveWidget=(a,i)=>void Kt(e,n,{name:a,decision:i}),o.actions=Lc(n,t.operator===!0)),o}function Lc(t,e){let r={invoke:async s=>{let n=await t.request("dashboard.action.invoke",s);return k(n)&&n.pending===!0?{kind:"pending",id:typeof n.id=="string"?n.id:"",expiresAt:typeof n.expiresAt=="string"?n.expiresAt:""}:{kind:"result",result:n}},subscribe:s=>t.addEventListener("dashboard.action.changed",n=>{k(n)&&typeof n.id=="string"&&s({id:n.id,status:n.status,connector:typeof n.connector=="string"?n.connector:"",tool:typeof n.tool=="string"?n.tool:""})})};return e&&(r.confirm=async s=>{let n=await t.request("dashboard.action.confirm",{id:s});return{result:k(n)&&"result"in n?n.result:n}},r.deny=async s=>{await t.request("dashboard.action.deny",{id:s})}),r}function Pc(t){return Object.entries(t.widgetsRegistry).filter(([,e])=>e.status==="pending").map(([e])=>e)}function Wc(t,e){let r=s=>s.sessionKey===e;return{send:async s=>({turnId:(await t.request("chat.send",{sessionKey:e,message:s})).turnId}),abort:async s=>{await t.request("chat.abort",{sessionKey:e,turnId:s})},history:async()=>((await t.request("chat.history.get",{sessionKey:e})).events??[]).filter(r),subscribe:s=>t.addEventListener(Oo,n=>{let o=n;o&&r(o)&&s(o)})}}function Uc(t,e){return{get:()=>t.request("dashboard.widget.state.get",{widgetId:e}),set:r=>t.request("dashboard.widget.state.set",{widgetId:e,state:r})}}function an(t,e,r,s,n,o){let a=Le(n.kind);return a?{status:vs(s,n.kind),manifest:r.manifestCache.get(a)??null,host:{transport:t.transport,basePath:t.basePath??"",sessionKey:t.sessionKey??"main",tabSlug:o,...t.confirm?{confirmPrompt:t.confirm}:{}},onApprove:()=>void Kt(e,t.transport,{name:a,decision:"approved"}),onReject:()=>void Kt(e,t.transport,{name:a,decision:"rejected"})}:null}function zc(t){return[...t.history.snapshots.entries()].map(([e,r])=>({version:e,workspace:r}))}function jc(t,e,r){let s=r.createdBy;if(!s)return;let n=X(s),o=zo(r.id,zc(e));return{actor:s,agentId:n,...o!==void 0?{firstSeenVersion:o}:{},...n?{logbookHref:t.logbookHref??null}:{}}}async function Fc(t,e){let r=()=>t.onRequestUpdate?.(),s=e.history;s.loading=!0,s.error=null,r();try{let n=await Rd(t.transport);s.entries=n,n.length>0&&s.selectedVersion===null&&(s.selectedVersion=n[0].version),s.error=null}catch(n){s.error=n instanceof Error?n.message:String(n)}finally{s.loading=!1,r()}s.selectedVersion!==null&&await dn(t,e,s.selectedVersion)}async function dn(t,e,r){let s=e.history;if(!s.snapshots.has(r))try{let n=await Id(t.transport,r);n&&(s.snapshots.set(r,n),t.onRequestUpdate?.())}catch(n){s.error=n instanceof Error?n.message:String(n),t.onRequestUpdate?.()}}function Hc(t,e){e.history.open=!0,e.history.confirmRestore=!1,Fc(t,e),t.onRequestUpdate?.()}function Se(t,e){e.history.open=!1,e.history.confirmRestore=!1,t.onRequestUpdate?.()}function Vc(t,e,r){e.history.selectedVersion=r,dn(t,e,r),t.onRequestUpdate?.()}function qc(t,e){let r=e.manifest,s=r.preferredSize&&typeof r.preferredSize=="object"?r.preferredSize:{},n=Math.min(12,Math.max(1,Number(s.w)||6)),o=Math.max(1,Number(s.h)||4);return{x:0,y:(t?.widgets??[]).reduce((a,i)=>{let l=i.grid.y+i.grid.h;return l>a?l:a},0),w:n,h:o}}function Gc(t,e,r,s,n){if(Ec(r,t.transport,s,n,t.onRequestUpdate??null),Bc(r,t,s,n),n.widgets.length===0)return c`
      <div class="dashboard-empty dashboard-empty--tab" data-test-id="dashboard-empty-tab">
        <span class="dashboard-empty__icon" aria-hidden="true">${M.plus}</span>
        <div class="dashboard-empty__title">${d("dashboard.empty.tabTitle")}</div>
        <div class="dashboard-empty__sub">${d("dashboard.empty.tabSubtitle")}</div>
      </div>
    `;if(n.layout==="full")return Kc(t,e,r,s,n);let o=ln(t,e,r,n),a=$s(n.widgets),i=a*56+Math.max(0,a-1)*12,l=tn(s).length>=2;return c`
    <div class="dashboard-grid" style="min-height: ${i}px" data-test-id="dashboard-grid">
      ${n.widgets.map(u=>{let b=an(t,e,r,s,u,n.slug),h=jc(t,r,u),m=r.drag,w=m?.widgetId===u.id,v=w&&m.mode==="move"?`translate(${m.pointerDx}px, ${m.pointerDy}px)`:void 0,g=l&&u.createdBy?cc(u.createdBy,r.highlightedAgent):null;return lc({widget:u,binding:r.bindingResults.get(u.id)??null,...h?{blame:h}:{},menuOpen:r.openMenuWidgetId===u.id,pending:e.pendingWidgetIds.has(u.id),dragging:w,...v?{dragTransform:v}:{},builtinContext:on(t,e,s,u),callbacks:o,...b?{custom:b}:{},...g?{agentChip:g}:{}})})}
      ${Jc(r,n)}
    </div>
  `}function Kc(t,e,r,s,n){let o=n.widgets[0],a=ln(t,e,r,n),i=an(t,e,r,s,o,n.slug);return c`
    <div class="dashboard-fullbleed" data-test-id="dashboard-fullbleed" data-widget-id=${o.id}>
      ${Ys(o,r.bindingResults.get(o.id)??null,on(t,e,s,o),a,i??void 0)}
    </div>
  `}function Jc(t,e){let r=t.drag;return r?c`
    <div
      class="dashboard-ghost ${We(r.ghostRect,e.widgets,r.widgetId)?"dashboard-ghost--invalid":""}"
      style=${Ue(r.ghostRect)}
      aria-hidden="true"
      data-test-id="dashboard-drag-ghost"
    ></div>
  `:p}function ln(t,e,r,s){let n=()=>t.onRequestUpdate?.(),o=(a,i,l)=>{let u=Tc(t.host);if(u.width<=0)return;let b=da({widget:a,mode:l,clientX:i.clientX,clientY:i.clientY,metrics:u});r.drag=b;let h=i.target;try{h.setPointerCapture?.(i.pointerId)}catch{}let m=!1,w=()=>{window.removeEventListener("pointermove",g),window.removeEventListener("pointerup",_)},v=()=>{m||(m=!0,w(),r.drag=null,n())},g=$=>{la(b,$.clientX,$.clientY),n()},_=()=>{if(m)return;m=!0,w(),Zi(t.host);let $=Rr({requested:b.ghostRect,widgets:s.widgets,widgetId:a.id});r.drag=null,n(),$&&($.x!==a.grid.x||$.y!==a.grid.y||$.w!==a.grid.w||$.h!==a.grid.h)&&Lr(e,t.transport,{slug:s.slug,widgetId:a.id,grid:$})};window.addEventListener("pointermove",g),window.addEventListener("pointerup",_),Yi(t.host,v)};return{onToggleCollapse:a=>void od(e,t.transport,{slug:s.slug,widgetId:a.id,collapsed:!a.collapsed}),onToggleMenu:a=>{r.openMenuWidgetId=r.openMenuWidgetId===a.id?null:a.id,n()},onHide:a=>{r.openMenuWidgetId=null,dd(e,t.transport,{slug:s.slug,widgetId:a.id})},onRemove:a=>{r.openMenuWidgetId=null,ld(e,t.transport,{slug:s.slug,widgetId:a.id})},onEditTitle:a=>{r.openMenuWidgetId=null,r.dialog={kind:"editTitle",slug:s.slug,widgetId:a.id,title:a.title},n()},onMoveToTab:a=>{r.openMenuWidgetId=null,r.dialog={kind:"moveToTab",slug:s.slug,widgetId:a.id},n()},onPin:a=>{r.openMenuWidgetId=null,id(e,t.transport,{slug:s.slug,widgetId:a.id})},onMovePointerDown:(a,i)=>{i.button===0&&(i.preventDefault(),o(a,i,"move"))},onResizePointerDown:(a,i)=>{i.button===0&&(i.preventDefault(),i.stopPropagation(),o(a,i,"resize"))},onKeyboardNudge:(a,i,l)=>{let u=Rr({requested:ua(a.grid,i,l),widgets:s.widgets,widgetId:a.id});u&&Lr(e,t.transport,{slug:s.slug,widgetId:a.id,grid:u})}}}function te(t,e,r){return c`
    <div
      class="bs-modal"
      role="dialog"
      aria-modal="true"
      aria-label=${t}
      data-test-id="bs-modal"
      @click=${o=>{o.target===o.currentTarget&&e()}}
      @keydown=${o=>{o.key==="Escape"&&(o.preventDefault(),e())}}
    >
      <div class="bs-modal__card">${r}</div>
    </div>
  `}function Xc(t,e,r){let s=r.dialog;if(!s)return p;let n=()=>t.onRequestUpdate?.(),o=()=>{r.dialog=null,n()};if(s.kind==="editTitle"){let u=d("dashboard.widget.editTitleTitle");return te(u,o,c`
        <form class="bs-dialog" @submit=${h=>{h.preventDefault();let m=h.currentTarget.querySelector("input[name='dashboard-widget-title']")?.value.trim()??"";m&&m!==s.title&&ad(e,t.transport,{slug:s.slug,widgetId:s.widgetId,title:m}),o()}}>
          <div class="bs-dialog__title">${u}</div>
          <input
            class="bs-dialog__input"
            type="text"
            name="dashboard-widget-title"
            data-test-id="dashboard-edit-title-input"
            .value=${s.title}
            aria-label=${d("dashboard.widget.editTitleLabel")}
          />
          <div class="bs-dialog__actions">
            <button class="bs-btn bs-btn--primary" type="submit">${d("common.save")}</button>
            <button class="bs-btn" type="button" @click=${o}>${d("common.cancel")}</button>
          </div>
        </form>
      `)}let a=d("dashboard.widget.moveToTabTitle"),i=e.workspace?ne(e.workspace).filter(u=>u.slug!==s.slug):[];return te(a,o,c`
      <form class="bs-dialog" @submit=${u=>{u.preventDefault();let b=u.currentTarget.querySelector("select[name='dashboard-move-target']")?.value??"";b&&b!==s.slug&&cd(e,t.transport,{fromSlug:s.slug,toSlug:b,widgetId:s.widgetId}),o()}}>
        <div class="bs-dialog__title">${a}</div>
        ${i.length===0?c`<div class="bs-dialog__sub">${d("dashboard.widget.moveToTabEmpty")}</div>`:c`<select
                class="bs-dialog__input"
                name="dashboard-move-target"
                data-test-id="dashboard-move-target"
                aria-label=${a}
              >
                ${i.map(u=>c`<option value=${u.slug}>${u.title}</option>`)}
              </select>`}
        <div class="bs-dialog__actions">
          <button class="bs-btn bs-btn--primary" type="submit" ?disabled=${i.length===0}>
            ${d("dashboard.widget.menu.moveToTab")}
          </button>
          <button class="bs-btn" type="button" @click=${o}>${d("common.cancel")}</button>
        </div>
      </form>
    `)}function Yc(t){qd(t.strings);let e=td(t.host),r=_c(t.host,t.storage);e.requestUpdate=t.onRequestUpdate??null,yc(t.host,r,()=>t.onRequestUpdate?.());let s=t.connected;return rd(t.host,e,s?t.transport:null),sd(t.host,s?t.transport:null,()=>{xc(t.host),s&&e.activeSlug&&Br(t.host,t.transport,e.activeSlug),t.onRequestUpdate?.()}),s&&!e.loaded&&!e.loading&&!e.error&&dt(e,t.transport,{requestedSlug:t.initialTab??null}),s&&e.activeSlug&&r.lastPresenceSlug!==e.activeSlug&&(r.lastPresenceSlug=e.activeSlug,Br(t.host,t.transport,e.activeSlug)),c`
    <section class="dashboard" data-test-id="dashboard">
      ${e.actionError?c`<div class="callout danger dashboard__toast" role="alert">
              ${e.actionError}
            </div>`:p}
      ${Zc(t,e,r)} ${Xc(t,e,r)}
      ${au(t,e,r)} ${gu(t,e,r)}
    </section>
  `}function Zc(t,e,r){if(e.error)return c`
      <div class="card lazy-view-state" role="alert">
        <div class="card-title">${d("dashboard.error.title")}</div>
        <div class="card-sub">${d("dashboard.error.subtitle")}</div>
        <details class="dashboard-error-detail">
          <summary>${d("dashboard.error.detailSummary")}</summary>
          <div class="dashboard-error-detail__text">${e.error}</div>
        </details>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          @click=${()=>void dt(e,t.transport)}
        >
          ${d("common.reload")}
        </button>
      </div>
    `;let s=e.workspace;if(!s)return c`
      <div class="dashboard-skeleton" role="status" aria-label=${d("common.loading")}>
        ${[0,1,2,3,4,5].map(()=>c`<div class="dashboard-skeleton__card"></div>`)}
      </div>
    `;if(s.tabs.length===0)return c`
      <div class="dashboard-empty dashboard-empty--onboarding" data-test-id="dashboard-empty">
        <div class="dashboard-empty__title">${d("dashboard.empty.onboardingTitle")}</div>
        <div class="dashboard-empty__sub">${d("dashboard.empty.onboardingSubtitle")}</div>
        <code class="dashboard-empty__cmd">${d("dashboard.empty.onboardingCommand")}</code>
      </div>
    `;let n=Pe(s,e.activeSlug)??oe(s)[0];return n?c`
    ${nu(t,e,r,n)}
    ${Ic(t,r,s,()=>t.onRequestUpdate?.())}
    ${Oc(t,e,r,s)}
    ${Qc(t,r,s)}
    ${Gc(t,e,r,s,n)}
  `:c`<div class="card lazy-view-state" role="status">
      <div class="card-sub">${d("dashboard.empty.noVisibleTabs")}</div>
    </div>`}function Qc(t,e,r){let s=tn(r);if(s.length<2)return e.highlightedAgent=null,p;let n=a=>{e.highlightedAgent=e.highlightedAgent===a?null:a,t.onRequestUpdate?.()},o=e.highlightedAgent;return c`
    <div
      class="dashboard-agent-filter"
      data-test-id="dashboard-agent-filter"
      role="group"
      aria-label=${d("dashboard.agentFilter.label")}
    >
      <span class="dashboard-agent-filter__label">${d("dashboard.agentFilter.label")}</span>
      <button
        class="dashboard-agent-filter__chip ${o===null?"dashboard-agent-filter__chip--active":""}"
        type="button"
        data-test-id="dashboard-agent-filter-all"
        aria-pressed=${o===null?"true":"false"}
        @click=${()=>n(null)}
      >
        ${d("dashboard.agentFilter.all")}
      </button>
      ${s.map(a=>{let i=X(a)??a,l=o===a;return c`<button
          class="dashboard-agent-filter__chip dashboard-agent-filter__chip--agent ${l?"dashboard-agent-filter__chip--active":""}"
          type="button"
          style="--dashboard-agent-hue: ${Zs(a)}"
          data-agent=${a}
          data-test-id="dashboard-agent-filter-chip"
          aria-pressed=${l?"true":"false"}
          title=${d("dashboard.widget.agentChipTooltip",{agent:a})}
          @click=${()=>n(a)}
        >
          ${Qs(i)}
        </button>`})}
    </div>
  `}function tu(t,e){if(typeof document>"u"||typeof URL.createObjectURL!="function")return;let r=new Blob([e],{type:"application/json"}),s=URL.createObjectURL(r),n=document.createElement("a");n.href=s,n.download=t,document.body.append(n),n.click(),n.remove(),URL.revokeObjectURL(s)}function eu(t,e){pd(t.transport).then(r=>tu(r.filename,r.json)).catch(r=>{e.actionError=r instanceof Error?r.message:String(r),t.onRequestUpdate?.()})}function ru(t,e,r){let s=r.currentTarget,n=s.files?.[0];s.value="",n&&n.text().then(o=>gd(e,t.transport,o))}function su(t,e){e.gallery={indexUrl:pc(t.storage),mode:"widgets",entries:null,selected:null,recipes:null,selectedRecipe:null,busy:!1,error:null},t.onRequestUpdate?.()}function nu(t,e,r,s){let n=s.layout==="full",o=()=>void ud(e,t.transport,{slug:s.slug,layout:n?"grid":"full"});return c`
    <div class="dashboard-page-header" data-test-id="dashboard-page-header">
      <div class="dashboard-page-header__titles">
        <div class="page-title">${s.title}</div>
        <div class="page-sub">${d("dashboard.header.subtitle")}</div>
      </div>
      <div
        class="dashboard-page-header__actions dashboard-toolbar"
        data-test-id="dashboard-toolbar"
      >
        <button
          class="bs-btn bs-btn--small"
          type="button"
          data-test-id="dashboard-gallery-open"
          title=${d("dashboard.gallery.open")}
          @click=${()=>su(t,r)}
        >
          <span class="dashboard-page-header__action-icon" aria-hidden="true">${M.puzzle}</span>
          ${d("dashboard.gallery.open")}
        </button>
        <button
          class="bs-btn bs-btn--small ${n?"bs-btn--primary":""}"
          type="button"
          data-test-id="dashboard-fullbleed-toggle"
          aria-pressed=${n?"true":"false"}
          title=${d(n?"dashboard.header.fullBleedExit":"dashboard.header.fullBleedEnter")}
          @click=${o}
        >
          <span class="dashboard-page-header__action-icon" aria-hidden="true"
            >${n?M.minimize:M.maximize}</span
          >
          ${d(n?"dashboard.header.fullBleedExit":"dashboard.header.fullBleedEnter")}
        </button>
        <button
          class="bs-btn bs-btn--small dashboard-history__toggle"
          type="button"
          data-test-id="dashboard-history-toggle"
          @click=${()=>Hc(t,r)}
        >
          ${M.clock} ${d("dashboard.history.open")}
        </button>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          data-test-id="dashboard-export"
          title=${d("dashboard.distribution.exportTitle")}
          @click=${()=>eu(t,e)}
        >
          ${d("dashboard.distribution.export")}
        </button>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          data-test-id="dashboard-import"
          title=${d("dashboard.distribution.importTitle")}
          @click=${a=>a.currentTarget.parentElement?.querySelector('input[type="file"]')?.click()}
        >
          ${d("dashboard.distribution.import")}
        </button>
        <input
          type="file"
          accept="application/json,.json"
          hidden
          data-test-id="dashboard-import-input"
          @change=${a=>ru(t,e,a)}
        />
      </div>
    </div>
  `}function ou(t){let e=Date.parse(t);if(!Number.isFinite(e))return t;let r=Math.round((Date.now()-e)/1e3);if(r<60)return"just now";let s=Math.round(r/60);if(s<60)return`${s}m ago`;let n=Math.round(s/60);if(n<24)return`${n}h ago`;let o=Math.round(n/24);if(o<7)return`${o}d ago`;try{return new Intl.DateTimeFormat("en-US",{month:"short",day:"numeric"}).format(new Date(e))}catch{return t}}function au(t,e,r){let s=r.history;if(!s.open)return p;let n=d("dashboard.history.title"),o=s.selectedVersion!==null?s.snapshots.get(s.selectedVersion):void 0,a=s.entries[0]?.version??null;return te(n,()=>Se(t,r),c`
      <div class="dashboard-history" data-test-id="dashboard-history">
        <div class="dashboard-history__header">
          <div class="card-title">${n}</div>
          <div class="card-sub">${d("dashboard.history.subtitle")}</div>
        </div>
        ${s.error?c`<div class="callout danger" role="alert">${s.error}</div>`:p}
        <div class="dashboard-history__body">
          ${lu(t,r,a)}
          <div class="dashboard-history__detail">
            ${s.selectedVersion===null?c`<div class="card-sub">${d("dashboard.history.emptyDetail")}</div>`:cu(t,e,r,s.selectedVersion,o)}
          </div>
        </div>
        <div class="bs-dialog__actions">
          <button class="bs-btn" type="button" @click=${()=>Se(t,r)}>
            ${d("common.close")}
          </button>
        </div>
      </div>
    `)}function iu(t){let e=[];return t.added>0&&e.push(d("dashboard.history.summary.added",{count:String(t.added)})),t.removed>0&&e.push(d("dashboard.history.summary.removed",{count:String(t.removed)})),t.moved>0&&e.push(d("dashboard.history.summary.moved",{count:String(t.moved)})),t.retitled>0&&e.push(d("dashboard.history.summary.retitled",{count:String(t.retitled)})),t.tabsChanged>0&&e.push(d("dashboard.history.summary.tabs",{count:String(t.tabsChanged)})),e.length>0?e.join(" \xB7 "):d("dashboard.history.summary.minor")}function du(t){return t?c`<span class="dashboard-history__change">
    <span class="dashboard-history__change-label">${iu(t)}</span>
  </span>`:p}function lu(t,e,r){let s=e.history;return s.loading&&s.entries.length===0?c`<div class="dashboard-history__list">
      <div class="card-sub">${d("common.loading")}</div>
    </div>`:s.entries.length===0?c`<div class="dashboard-history__list">
      <div class="card-sub">${d("dashboard.history.empty")}</div>
    </div>`:c`
    <ul class="dashboard-history__list" role="listbox" aria-label=${d("dashboard.history.title")}>
      ${s.entries.map(n=>{let o=n.version===s.selectedVersion;return c`
          <li>
            <button
              class="dashboard-history__item ${o?"dashboard-history__item--active":""}"
              type="button"
              role="option"
              aria-selected=${o?"true":"false"}
              data-test-id="dashboard-history-item"
              @click=${()=>Vc(t,e,n.version)}
            >
              <span class="dashboard-history__version"
                >${d("dashboard.history.version",{version:String(n.version)})}</span
              >
              ${du(n.summary)}
              <span class="dashboard-history__time">${ou(n.savedAt)}</span>
              ${n.version===r?c`<span class="dashboard-history__latest"
                      >${d("dashboard.history.latest")}</span
                    >`:p}
            </button>
          </li>
        `})}
    </ul>
  `}function cu(t,e,r,s,n){let o=r.history,a=e.workspace,i=s===(o.entries[0]?.version??null);return n?c`
    <div class="dashboard-history__preview-wrap">
      <div class="dashboard-history__section-title">${d("dashboard.history.previewTitle")}</div>
      ${hu(n,e.activeSlug,s)}
    </div>
    <div class="dashboard-history__diff">
      <div class="dashboard-history__section-title">${d("dashboard.history.diffTitle")}</div>
      ${a?pu(n,a):p}
    </div>
    <div class="dashboard-history__restore">
      ${i?o.confirmRestore?c`
                <span class="dashboard-history__confirm"
                  >${d("dashboard.history.restoreConfirm")}</span
                >
                <button
                  class="bs-btn bs-btn--small bs-btn--primary"
                  type="button"
                  ?disabled=${o.restoring}
                  data-test-id="dashboard-history-restore-confirm"
                  @click=${async()=>{o.restoring=!0,t.onRequestUpdate?.(),await bd(e,t.transport),o.restoring=!1,o.confirmRestore=!1,Se(t,r)}}
                >
                  ${d("dashboard.history.restore")}
                </button>
                <button
                  class="bs-btn bs-btn--small"
                  type="button"
                  @click=${()=>{o.confirmRestore=!1,t.onRequestUpdate?.()}}
                >
                  ${d("common.cancel")}
                </button>
              `:c`<button
                class="bs-btn bs-btn--small"
                type="button"
                data-test-id="dashboard-history-restore"
                @click=${()=>{o.confirmRestore=!0,t.onRequestUpdate?.()}}
              >
                ${d("dashboard.history.restore")}
              </button>`:c`<span class="card-sub">${d("dashboard.history.restoreOnlyNewest")}</span>`}
    </div>
  `:c`<div class="card-sub" data-test-id="dashboard-history-loading">
      ${d("common.loading")}
    </div>`}var Yr={chart:x`<polyline points="3 15 8 10 12 13 17 6 21 9" /><path d="M3 20h18" opacity="0.5" />`,"stat-card":x`<path d="M4 8h9" stroke-width="2.6" /><path d="M4 14h6" opacity="0.6" />`,table:x`<rect x="3" y="5" width="18" height="14" rx="1.5" /><path d="M3 10h18M3 15h18M9 5v14" opacity="0.6" />`,markdown:x`<path d="M4 7h16M4 12h16M4 17h9" opacity="0.85" />`,notes:x`<path d="M5 6h11M5 11h11M5 16h7" opacity="0.8" /><path d="M16 15l3-3 2 2-3 3-2 1z" />`,list:x`<circle cx="5" cy="7" r="1" /><circle cx="5" cy="12" r="1" /><circle cx="5" cy="17" r="1" /><path d="M9 7h11M9 12h11M9 17h7" opacity="0.8" />`,gauge:x`<path d="M4 16a8 8 0 0 1 16 0" /><path d="M12 16l4-3" />`,button:x`<rect x="4" y="9" width="16" height="6" rx="3" />`,frame:x`<rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 8h18" opacity="0.6" />`,custom:x`<path
    d="M4 7h3a1.5 1.5 0 1 0 3 0h3v3a1.5 1.5 0 1 1 0 3v3h-3a1.5 1.5 0 1 0-3 0H4v-3a1.5 1.5 0 1 1 0-3z"
  />`,default:x`<rect x="4" y="5" width="16" height="14" rx="2" opacity="0.6" />`},uu={activity:"list","agent-status":"list",approvals:"list",sessions:"list",instances:"list",cron:"list",chat:"list",usage:"gauge","action-button":"button","action-form":"button","iframe-embed":"frame",preview:"frame"};function bu(t){let e=t.startsWith("custom:")?"custom":t.replace(/^builtin:/,"");return c`<svg
    class="dashboard-history__cell-glyph"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    ${Yr[e]??Yr[uu[e]??"default"]}
  </svg>`}function hu(t,e,r){let s=(e?t.tabs.find(o=>o.slug===e):void 0)??oe(t)[0]??t.tabs[0];if(!s||s.widgets.length===0)return c`<div class="dashboard-history__preview dashboard-history__preview--empty">
      ${d("dashboard.history.previewEmpty")}
    </div>`;let n=$s(s.widgets);return c`
    <div
      class="dashboard-history__preview dashboard-grid dashboard-grid--readonly"
      style="min-height: ${n*56+Math.max(0,n-1)*12}px"
      data-test-id="dashboard-history-preview"
      aria-hidden="true"
    >
      ${s.widgets.map(o=>{let a=X(o.createdBy);return c`
          <div class="dashboard-history__cell" style=${Ue(o.grid)}>
            ${bu(o.kind)}
            <span class="dashboard-history__cell-title">${o.title||o.kind}</span>
            ${a?c`<span class="dashboard-widget__provenance"
                    >${d("dashboard.widget.provenanceChip")}</span
                  >`:p}
          </div>
        `})}
    </div>
    <div class="dashboard-history__preview-caption">
      ${d("dashboard.history.previewCaption",{version:String(r)})}
    </div>
  `}function pu(t,e){let r=Po(t,e);return r.length===0?c`<div class="card-sub" data-test-id="dashboard-history-diff-empty">
      ${d("dashboard.history.diffEmpty")}
    </div>`:c`
    <div class="dashboard-history__diff-groups" data-test-id="dashboard-history-diff">
      ${Wo(r).map(s=>c`
          <div class="dashboard-history__diff-group">
            <div class="dashboard-history__diff-actor">
              ${s.actor??d("dashboard.history.actorUnknown")}
            </div>
            <ul class="dashboard-history__diff-list">
              ${s.entries.map(n=>c`
                  <li class="dashboard-history__diff-item">
                    <span class="dashboard-history__diff-kind"
                      >${d(`dashboard.history.kind.${n.kind}`)}</span
                    >
                    <span class="dashboard-history__diff-label">${n.label}</span>
                    ${n.detail?c`<span class="dashboard-history__diff-detail">${n.detail}</span>`:p}
                  </li>
                `)}
            </ul>
          </div>
        `)}
    </div>
  `}function gu(t,e,r){let s=r.gallery;if(!s)return p;let n=()=>t.onRequestUpdate?.(),o=()=>{r.gallery=null,n()},a=g=>{s.indexUrl=g.currentTarget.value},i=g=>{s.mode=g,s.selected=null,s.selectedRecipe=null,s.error=null,n()},l=async()=>{let g=s.indexUrl.trim();if(g){s.busy=!0,s.error=null,s.selected=null,s.selectedRecipe=null,n();try{let[_,$]=await Promise.all([Nd(g),Cd(g)]);s.entries=_,s.recipes=$,gc(t.storage,g)}catch(_){s.error=bt(_)}finally{s.busy=!1,n()}}},u=async g=>{s.busy=!0,s.error=null,n();try{s.selected=await Md(g.manifestUrl)}catch(_){s.error=bt(_)}finally{s.busy=!1,n()}},b=async g=>{s.busy=!0,s.error=null,n();try{s.selectedRecipe=await Od(g.manifestUrl)}catch(_){s.error=bt(_)}finally{s.busy=!1,n()}},h=async()=>{let g=s.selectedRecipe;if(g){s.busy=!0,s.error=null,n();try{if(!await fd(e,t.transport,g)){s.error=e.actionError??bt(new Error("Install failed.")),s.busy=!1,n();return}let _=g.doc.tabs[0]?.slug;_&&(e.activeSlug=_,t.onNavigate?.(_)),r.gallery=null,n()}catch(_){s.error=bt(_),s.busy=!1,n()}}},m=async()=>{let g=s.selected;if(g){s.busy=!0,s.error=null,n();try{await Bd(t.transport,g);let _=e.workspace?Pe(e.workspace,e.activeSlug):void 0;t.transport&&_&&await t.transport.request("dashboard.widget.add",{tab:_.slug,widget:{kind:`custom:${g.name}`,title:g.title,grid:qc(_,g)}}),await dt(e,t.transport,{silent:!0}),r.gallery=null,n()}catch(_){s.error=bt(_),s.busy=!1,n()}}},w=()=>s.selected?mu(s.selected,()=>{s.selected=null,n()},()=>void m(),s.busy):fu(s,g=>void u(g)),v=()=>s.selectedRecipe?wu(s.selectedRecipe,()=>{s.selectedRecipe=null,n()},()=>void h(),s.busy):yu(s,g=>void b(g));return te(d("dashboard.gallery.title"),o,c`
      <div class="dashboard-gallery" data-test-id="dashboard-gallery">
        <div class="dashboard-gallery__header">
          <div class="card-title">${d("dashboard.gallery.title")}</div>
          <div class="card-sub">${d("dashboard.gallery.subtitle")}</div>
        </div>
        <div class="dashboard-gallery__tabs" role="tablist">
          <button
            class="dashboard-gallery__tab ${s.mode==="widgets"?"is-active":""}"
            type="button"
            role="tab"
            aria-selected=${s.mode==="widgets"}
            data-test-id="dashboard-gallery-tab-widgets"
            @click=${()=>i("widgets")}
          >
            ${d("dashboard.gallery.tabWidgets")}
          </button>
          <button
            class="dashboard-gallery__tab ${s.mode==="templates"?"is-active":""}"
            type="button"
            role="tab"
            aria-selected=${s.mode==="templates"}
            data-test-id="dashboard-gallery-tab-templates"
            @click=${()=>i("templates")}
          >
            ${d("dashboard.gallery.tabTemplates")}
          </button>
        </div>
        <div class="dashboard-gallery__browse">
          <input
            class="bs-dialog__input"
            type="url"
            inputmode="url"
            data-test-id="dashboard-gallery-url"
            placeholder=${d("dashboard.gallery.urlPlaceholder")}
            aria-label=${d("dashboard.gallery.urlLabel")}
            .value=${s.indexUrl}
            @input=${a}
          />
          <button
            class="bs-btn bs-btn--small bs-btn--primary"
            type="button"
            data-test-id="dashboard-gallery-browse"
            ?disabled=${s.busy}
            @click=${()=>void l()}
          >
            ${d("dashboard.gallery.browse")}
          </button>
        </div>
        ${s.error?c`<div class="callout danger" role="alert" data-test-id="dashboard-gallery-error">
                ${s.error}
              </div>`:p}
        ${s.mode==="templates"?v():w()}
      </div>
    `)}function fu(t,e){return t.entries===null?p:t.entries.length===0?c`<div class="dashboard-gallery__empty">${d("dashboard.gallery.empty")}</div>`:c`
    <ul class="dashboard-gallery__list" data-test-id="dashboard-gallery-list">
      ${t.entries.map(r=>c`
          <li class="dashboard-gallery__item">
            <div class="dashboard-gallery__item-body">
              <div class="dashboard-gallery__item-name">${r.name}</div>
              ${r.description?c`<div class="dashboard-gallery__item-desc">${r.description}</div>`:p}
            </div>
            <button
              class="bs-btn bs-btn--small"
              type="button"
              data-test-id="dashboard-gallery-select"
              ?disabled=${t.busy}
              @click=${()=>e(r)}
            >
              ${d("dashboard.gallery.view")}
            </button>
          </li>
        `)}
    </ul>
  `}function mu(t,e,r,s){return c`
    <div class="dashboard-gallery__detail" data-test-id="dashboard-gallery-detail">
      <div class="dashboard-gallery__item-name">${t.title}</div>
      <div class="dashboard-gallery__caps">
        <div class="dashboard-gallery__caps-label">${d("dashboard.gallery.capabilities")}</div>
        ${t.capabilities.length===0?c`<span class="dashboard-gallery__cap"
                >${d("dashboard.gallery.noCapabilities")}</span
              >`:t.capabilities.map(n=>c`<span class="dashboard-gallery__cap" data-test-id="dashboard-gallery-cap"
                    >${n}</span
                  >`)}
      </div>
      <div class="dashboard-gallery__pending-note">${d("dashboard.gallery.pendingNote")}</div>
      <div class="bs-dialog__actions">
        <button
          class="bs-btn bs-btn--primary"
          type="button"
          data-test-id="dashboard-gallery-install"
          ?disabled=${s}
          @click=${r}
        >
          ${d("dashboard.gallery.install")}
        </button>
        <button class="bs-btn" type="button" @click=${e}>${d("common.back")}</button>
      </div>
    </div>
  `}function yu(t,e){return t.recipes===null?p:t.recipes.length===0?c`<div class="dashboard-gallery__empty">${d("dashboard.gallery.recipesEmpty")}</div>`:c`
    <ul class="dashboard-gallery__list" data-test-id="dashboard-gallery-recipe-list">
      ${t.recipes.map(r=>c`
          <li class="dashboard-gallery__item">
            <div class="dashboard-gallery__item-body">
              <div class="dashboard-gallery__item-name">${r.title}</div>
              ${r.description?c`<div class="dashboard-gallery__item-desc">${r.description}</div>`:p}
              <div class="dashboard-gallery__recipe-needs">
                ${r.connectors.length===0?d("dashboard.gallery.recipeNeedsNothing"):d("dashboard.gallery.recipeNeedsConnectors",{connectors:r.connectors.join(", ")})}
              </div>
            </div>
            <button
              class="bs-btn bs-btn--small"
              type="button"
              data-test-id="dashboard-gallery-recipe-select"
              ?disabled=${t.busy}
              @click=${()=>e(r)}
            >
              ${d("dashboard.gallery.view")}
            </button>
          </li>
        `)}
    </ul>
  `}function wu(t,e,r,s){let n=Object.entries(t.grantsManifest);return c`
    <div class="dashboard-gallery__detail" data-test-id="dashboard-gallery-recipe-detail">
      <div class="dashboard-gallery__item-name">${t.title}</div>
      <div class="dashboard-gallery__item-desc">${t.description}</div>
      <div class="dashboard-gallery__recipe-grants">
        <div class="dashboard-gallery__caps-label">${d("dashboard.gallery.recipeNeedsLabel")}</div>
        ${n.length===0?c`<div class="dashboard-gallery__recipe-nogrants">
                ${d("dashboard.gallery.recipeNoGrants")}
              </div>`:n.map(([,o])=>c`
                  <div class="dashboard-gallery__recipe-connector">
                    <div class="dashboard-gallery__recipe-connector-name">${o.label}</div>
                    ${o.reason?c`<div class="dashboard-gallery__recipe-connector-reason">
                            ${o.reason}
                          </div>`:p}
                    <ul class="dashboard-gallery__recipe-tools">
                      ${(o.tools??[]).map(a=>c`
                          <li
                            class="dashboard-gallery__recipe-tool"
                            data-test-id="dashboard-gallery-recipe-tool"
                          >
                            <code>${a.id}</code>
                            <span>${a.label}</span>
                            ${a.readOnly?c`<span class="dashboard-gallery__recipe-readonly"
                                    >${d("dashboard.gallery.recipeReadOnly")}</span
                                  >`:p}
                          </li>
                        `)}
                    </ul>
                  </div>
                `)}
      </div>
      <div class="dashboard-gallery__pending-note">${d("dashboard.gallery.recipeInstallNote")}</div>
      <div class="bs-dialog__actions">
        <button
          class="bs-btn bs-btn--primary"
          type="button"
          data-test-id="dashboard-gallery-recipe-install"
          ?disabled=${s}
          @click=${r}
        >
          ${d("dashboard.gallery.recipeInstall")}
        </button>
        <button class="bs-btn" type="button" @click=${e}>${d("common.back")}</button>
      </div>
    </div>
  `}var wt,vu=(wt=class extends ft{constructor(...e){super(...e),this.transport=null,this.connected=!1,this.operator=!1}createRenderRoot(){return this}render(){return Yc({host:this,transport:this.transport,connected:this.connected,onRequestUpdate:()=>this.requestUpdate(),...this.strings?{strings:this.strings}:{},...this.onNavigate?{onNavigate:this.onNavigate}:{},...this.storage?{storage:this.storage}:{},...this.confirm?{confirm:this.confirm}:{},...this.embed?{embed:this.embed}:{},...this.basePath!==void 0?{basePath:this.basePath}:{},...this.initialTab!==void 0?{initialTab:this.initialTab}:{},...this.sessionKey!==void 0?{sessionKey:this.sessionKey}:{},...this.logbookHref!==void 0?{logbookHref:this.logbookHref}:{},operator:this.operator})}disconnectedCallback(){super.disconnectedCallback(),nd(this),wc(this)}},wt.properties={transport:{attribute:!1},connected:{type:Boolean},strings:{attribute:!1},onNavigate:{attribute:!1},storage:{attribute:!1},confirm:{attribute:!1},embed:{attribute:!1},basePath:{type:String},initialTab:{type:String},sessionKey:{type:String},logbookHref:{type:String},operator:{type:Boolean}},wt);typeof customElements<"u"&&!customElements.get("boardstate-view")&&customElements.define("boardstate-view",vu);var vt,_u=(vt=class extends ft{constructor(...e){super(...e),this.currentLabel="",this.agentLabel="",this.brandLabel="",this.overviewHref="",this.handleOverviewClick=r=>{r.defaultPrevented||r.button!==0||r.metaKey||r.ctrlKey||r.shiftKey||r.altKey||(r.preventDefault(),this.dispatchEvent(new CustomEvent("navigate",{detail:"overview",bubbles:!0,composed:!0})))}}createRenderRoot(){return this}render(){let e=this.currentLabel.trim(),r=this.agentLabel.trim(),s=this.brandLabel.trim();return c`
      <div class="dashboard-header">
        <div class="dashboard-header__breadcrumb">
          ${s?this.overviewHref?c`<a
                    class="dashboard-header__breadcrumb-link"
                    href=${this.overviewHref}
                    @click=${this.handleOverviewClick}
                    >${s}</a
                  >`:c`<span class="dashboard-header__breadcrumb-link">${s}</span>`:p}
          ${r?c`
                  <span class="dashboard-header__breadcrumb-segment">
                    ${s?c`<span class="dashboard-header__breadcrumb-sep">›</span>`:p}
                    <span class="dashboard-header__breadcrumb-context" title=${r}>
                      ${r}
                    </span>
                  </span>
                `:p}
          ${e?c`
                  ${s||r?c`<span class="dashboard-header__breadcrumb-sep">›</span>`:p}
                  <span class="dashboard-header__breadcrumb-current">${e}</span>
                `:p}
        </div>
        <div class="dashboard-header__actions">
          <slot></slot>
        </div>
      </div>
    `}},vt.properties={currentLabel:{type:String},agentLabel:{type:String},brandLabel:{type:String},overviewHref:{type:String}},vt);typeof customElements<"u"&&!customElements.get("boardstate-header")&&customElements.define("boardstate-header",_u);var cn=`/*
 * @boardstate/lit reference view styles.
 *
 * Every themeable value is read as \`var(--bs-<token>, <default>)\` so the stylesheet
 * renders standalone yet is fully overridable: set the \`--bs-*\` custom properties on
 * any ancestor to theme it. See THEME.md for the full token table.
 *
 * The \`:root\` block below is the shipped default theme \u2014 "Graphite", a Linear/Vercel/
 * Codex-family palette that looks world-class light AND dark out of the box. Dark mode
 * activates two ways so either the OS preference or an explicit toggle drives it:
 *   - \`<html data-theme="dark">\` / \`data-theme="light">\` \u2014 explicit, always wins.
 *   - \`prefers-color-scheme: dark\` \u2014 used only when no explicit \`data-theme="light"\`.
 * Drop in an alternate theme (e.g. \`@boardstate/lit/themes/aurora.css\`) after this file
 * to fully re-skin, or override any \`--bs-*\` token on an ancestor.
 */

:root {
  /* Tells the UA to render form controls, scrollbars, etc. for the light theme. */
  color-scheme: light;

  /* Typography */
  --bs-font-sans: -apple-system, "SF Pro Text", system-ui, "Segoe UI", Roboto, sans-serif;
  --bs-font-mono: ui-monospace, "SF Mono", Menlo, monospace;

  /* Surfaces */
  --bs-bg: #fbfbfd;
  --bs-card: #ffffff;
  --bs-card-highlight: #f6f6fa;
  --bs-input: #ffffff;
  --bs-bg-hover: #f2f2f7;
  --bs-bg-muted: rgba(16, 16, 24, 0.03);

  /* Borders */
  --bs-border: #e7e7ee;
  --bs-border-strong: #d9d9e2;

  /* Text */
  --bs-text: #15151b;
  --bs-text-strong: #000000;
  --bs-text-muted: #6b6b77;
  --bs-text-dim: #9a9aa6;
  --bs-muted: #6b6b77;

  /* Accent / semantic \u2014 light values sit a step darker than dark mode's so
     accent-as-text and white-on-accent both clear WCAG AA (4.5:1) on white. */
  --bs-accent: #6c5bfa;
  --bs-accent-foreground: #ffffff;
  --bs-ring: rgba(108, 91, 250, 0.5);
  --bs-focus-ring: 0 0 0 2px rgba(108, 91, 250, 0.45);
  --bs-success: #27853c;
  --bs-warning: #986d0d;
  --bs-danger: #d92c25;
  --bs-danger-subtle: rgba(217, 44, 37, 0.1);

  /* Radii */
  --bs-radius-sm: 6px;
  --bs-radius-md: 9px;
  --bs-radius-lg: 12px;
  --bs-radius-full: 999px;

  /* Elevation / motion */
  --bs-shadow-md: 0 1px 2px rgba(16, 16, 24, 0.06), 0 6px 20px rgba(16, 16, 24, 0.08);
  --bs-ease-out: cubic-bezier(0.2, 0.8, 0.2, 1);
  --bs-duration-fast: 120ms;
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --bs-bg: #0b0b0f;
  --bs-card: #131318;
  --bs-card-highlight: #17171e;
  --bs-input: #17171e;
  --bs-bg-hover: rgba(255, 255, 255, 0.04);
  --bs-bg-muted: rgba(255, 255, 255, 0.02);
  --bs-border: #23232b;
  --bs-border-strong: #30303a;
  --bs-text: #ededf2;
  --bs-text-strong: #ffffff;
  --bs-text-muted: #9a9aa6;
  --bs-text-dim: #63636e;
  --bs-muted: #9a9aa6;
  --bs-accent: #7c6cff;
  --bs-accent-foreground: #ffffff;
  --bs-ring: rgba(124, 108, 255, 0.5);
  --bs-focus-ring: 0 0 0 2px rgba(124, 108, 255, 0.45);
  --bs-success: #3fb950;
  --bs-warning: #d29922;
  --bs-danger: #f85149;
  --bs-danger-subtle: rgba(248, 81, 73, 0.14);
  --bs-shadow-md: 0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px rgba(0, 0, 0, 0.28);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bs-bg: #0b0b0f;
    --bs-card: #131318;
    --bs-card-highlight: #17171e;
    --bs-input: #17171e;
    --bs-bg-hover: rgba(255, 255, 255, 0.04);
    --bs-bg-muted: rgba(255, 255, 255, 0.02);
    --bs-border: #23232b;
    --bs-border-strong: #30303a;
    --bs-text: #ededf2;
    --bs-text-strong: #ffffff;
    --bs-text-muted: #9a9aa6;
    --bs-text-dim: #63636e;
    --bs-muted: #9a9aa6;
    --bs-accent: #7c6cff;
    --bs-accent-foreground: #ffffff;
    --bs-ring: rgba(124, 108, 255, 0.5);
    --bs-focus-ring: 0 0 0 2px rgba(124, 108, 255, 0.45);
    --bs-success: #3fb950;
    --bs-warning: #d29922;
    --bs-danger: #f85149;
    --bs-danger-subtle: rgba(248, 81, 73, 0.14);
    --bs-shadow-md: 0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px rgba(0, 0, 0, 0.28);
  }
}

.dashboard {
  display: flex;
  flex-direction: column;
  gap: 12px;
  flex: 1;
  min-height: 0;
  padding-bottom: 16px;
  --dashboard-row-height: 56px;
  --dashboard-grid-gap: 12px;
  color: var(--bs-text, #1a1d21);
  font-family: var(--bs-font-sans, system-ui, sans-serif);
}

.dashboard__toast {
  margin: 0;
}

/* Generic surfaces the view borrows (were app-global in the source). */
.callout {
  padding: 10px 12px;
  border-radius: var(--bs-radius-md, 8px);
  border: 1px solid var(--bs-border, #e5e7eb);
  background: var(--bs-card, #fff);
}
.callout.danger {
  border-color: color-mix(in srgb, var(--bs-danger, #ef4444) 40%, var(--bs-border, #e5e7eb));
  background: color-mix(in srgb, var(--bs-danger, #ef4444) 8%, var(--bs-card, #fff));
  color: var(--bs-text, #1a1d21);
}
.card {
  padding: 16px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-lg, 12px);
  background: var(--bs-card, #fff);
}
.card-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--bs-text-strong, #111418);
}
.card-sub {
  font-size: 0.9em;
  color: var(--bs-text-muted, #6b7280);
}
.page-title {
  font-size: 20px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--bs-text-strong, #111418);
}
.page-sub {
  font-size: 0.9em;
  color: var(--bs-text-muted, #6b7280);
}

/* Local button (replaces the app's .btn). */
.bs-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 32px;
  padding: 0 12px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
  background: var(--bs-card, #fff);
  color: var(--bs-text, #1a1d21);
  font: inherit;
  cursor: pointer;
}
.bs-btn:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}
.bs-btn--small {
  height: 26px;
  padding: 0 10px;
  font-size: 0.9em;
}
.bs-btn--primary {
  background: var(--bs-accent, #6366f1);
  border-color: var(--bs-accent, #6366f1);
  color: var(--bs-accent-foreground, #fff);
}
.bs-btn--primary:hover {
  background: color-mix(in srgb, var(--bs-accent, #6366f1) 88%, #000);
}
.bs-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Local modal + dialog (replaces the app modal-dialog element). */
.bs-modal {
  position: fixed;
  inset: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: color-mix(in srgb, #000 60%, transparent);
  backdrop-filter: blur(3px);
  -webkit-backdrop-filter: blur(3px);
}
.bs-modal__card {
  width: auto;
  min-width: min(420px, calc(100vw - 48px));
  max-width: calc(100vw - 48px);
  max-height: calc(100vh - 48px);
  overflow: auto;
  overscroll-behavior: contain;
  padding: 16px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-lg, 12px);
  background: var(--bs-card, #fff);
  box-shadow: var(--bs-shadow-md, 0 8px 24px rgba(0, 0, 0, 0.18));
}
.bs-dialog {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.bs-dialog__title {
  font-size: 15px;
  font-weight: 600;
  color: var(--bs-text-strong, #111418);
}
.bs-dialog__sub {
  font-size: 0.9em;
  color: var(--bs-text-muted, #6b7280);
}
.bs-dialog__input {
  width: 100%;
  border: 1px solid var(--bs-input, var(--bs-border, #e5e7eb));
  background: var(--bs-card, #fff);
  border-radius: var(--bs-radius-md, 8px);
  padding: 8px 12px;
  color: var(--bs-text, #1a1d21);
  outline: none;
}
.bs-dialog__input:focus-visible {
  border-color: var(--bs-ring, var(--bs-accent, #6366f1));
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--bs-accent, #6366f1) 40%, transparent);
}
select.bs-dialog__input {
  appearance: none;
  cursor: pointer;
}
.bs-dialog__actions {
  display: flex;
  gap: 8px;
}

.dashboard-page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}

.dashboard-page-header__titles {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

/* Tab-level actions (gallery / full-bleed / history / export / import). */
.dashboard-page-header__actions,
.dashboard-toolbar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  flex-shrink: 0;
}

.dashboard-page-header__action-icon {
  display: inline-flex;
  width: 14px;
  height: 14px;
  margin-inline-end: 4px;
  vertical-align: -2px;
}

/* --- Tab strip ----------------------------------------------------------- */

.dashboard-tabs {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--bs-border, #e5e7eb);
  padding-bottom: 8px;
}
.dashboard-tab {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 32px;
  padding: 0 12px;
  border: 1px solid transparent;
  border-radius: var(--bs-radius-md, 8px);
  background: transparent;
  color: var(--bs-text-muted, #6b7280);
  font: inherit;
  cursor: pointer;
  transition:
    background 120ms var(--bs-ease-out, ease-out),
    color 120ms var(--bs-ease-out, ease-out);
}
.dashboard-tab:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
  color: var(--bs-text, #1a1d21);
}
.dashboard-tab--active {
  background: color-mix(in srgb, var(--bs-accent, #6366f1) 16%, transparent);
  border-color: color-mix(in srgb, var(--bs-accent, #6366f1) 40%, var(--bs-border, #e5e7eb));
  color: var(--bs-text, #1a1d21);
}
.dashboard-tab__icon svg {
  width: 15px;
  height: 15px;
}
.dashboard-tab__label {
  white-space: nowrap;
}
.dashboard-tabs__hidden {
  position: relative;
  margin-inline-start: auto;
}
.dashboard-tabs__hidden summary {
  list-style: none;
}
.dashboard-tabs__hidden summary::-webkit-details-marker {
  display: none;
}
.dashboard-tab--overflow {
  color: var(--bs-text-muted, #6b7280);
  flex-wrap: nowrap;
  white-space: nowrap;
}
.dashboard-tabs__hidden-menu {
  position: absolute;
  inset-inline-end: 4px;
  margin-top: 4px;
  min-width: 160px;
  display: flex;
  flex-direction: column;
  padding: 4px;
  background: var(--bs-card, #fff);
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
  box-shadow: var(--bs-shadow-md, 0 8px 24px rgba(0, 0, 0, 0.18));
  z-index: 20;
}
.dashboard-tabs__hidden-item {
  text-align: start;
  padding: 6px 8px;
  border: none;
  border-radius: var(--bs-radius-sm, 6px);
  background: transparent;
  color: var(--bs-text, #1a1d21);
  font: inherit;
  cursor: pointer;
}
.dashboard-tabs__hidden-item:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}

/* --- Grid ---------------------------------------------------------------- */

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  grid-auto-rows: var(--dashboard-row-height);
  gap: var(--dashboard-grid-gap);
  align-content: start;
  flex: 1;
  min-height: 0;
}

/* --- Widget cell --------------------------------------------------------- */

.dashboard-widget {
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
  position: relative;
  background: var(--bs-card, #fff);
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-lg, 12px);
}
.dashboard-widget--collapsed {
  grid-row: auto !important;
}
.dashboard-widget--pending {
  opacity: 0.7;
}
/* The landing-cell placeholder during a drag: deliberately QUIET \u2014 a neutral
   slot outline, not a colored signal \u2014 because the dragged card itself now
   carries the motion (Mac-style direct manipulation). Red stays reserved for
   an invalid (colliding) drop. */
.dashboard-ghost {
  pointer-events: none;
  border-radius: var(--bs-radius-lg, 12px);
  border: 2px dashed color-mix(in srgb, var(--bs-text-muted, #9a9aa6) 45%, transparent);
  background: color-mix(in srgb, var(--bs-text-muted, #9a9aa6) 7%, transparent);
  z-index: 5;
}
.dashboard-ghost--invalid {
  border-color: color-mix(in srgb, var(--bs-danger, #ef4444) 70%, transparent);
  background: color-mix(in srgb, var(--bs-danger, #ef4444) 10%, transparent);
}
.dashboard-widget--dragging {
  z-index: 10;
}
/* The carried card: lifted and following the pointer 1:1. No transition \u2014
   the transform must track the pointer with zero lag; the \u2264half-cell settle
   on drop is instant and imperceptible. */
/* Double class beats the later theme-section \`.dashboard-widget { box-shadow }\`. */
.dashboard-widget.dashboard-widget--carried {
  transition: none;
  box-shadow:
    0 18px 44px rgba(0, 0, 0, 0.38),
    0 4px 12px rgba(0, 0, 0, 0.24);
  opacity: 0.94;
  cursor: grabbing;
  will-change: transform;
}
.dashboard-widget--carried .dashboard-widget__bar {
  cursor: grabbing;
}
.dashboard-widget__bar {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: grab;
  padding-block: 4px;
  padding-inline: 4px 6px;
  border-bottom: 1px solid var(--bs-border, #e5e7eb);
  border-start-start-radius: var(--bs-radius-lg, 12px);
  border-start-end-radius: var(--bs-radius-lg, 12px);
  cursor: grab;
  touch-action: none;
  user-select: none;
}
.dashboard-widget__bar:active {
  cursor: grabbing;
}
.dashboard-widget__collapse,
.dashboard-widget__menu-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: none;
  border-radius: var(--bs-radius-sm, 6px);
  background: transparent;
  color: var(--bs-text-muted, #6b7280);
  cursor: pointer;
}
.dashboard-widget__collapse:hover,
.dashboard-widget__menu-toggle:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
  color: var(--bs-text, #1a1d21);
}
.dashboard-widget__menu-toggle[aria-expanded="true"] {
  background: color-mix(in srgb, var(--bs-accent, #6366f1) 16%, transparent);
  color: var(--bs-text, #1a1d21);
}
.dashboard-widget__collapse svg,
.dashboard-widget__menu-toggle svg {
  width: 15px;
  height: 15px;
}
.dashboard-widget__title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--bs-text-strong, #111418);
}
.dashboard-widget__provenance {
  display: inline-flex;
  align-items: center;
  padding: 1px 6px;
  border-radius: var(--bs-radius-full, 999px);
  background: color-mix(in srgb, var(--bs-accent, #6366f1) 18%, transparent);
  color: var(--bs-accent, #6366f1);
  font-size: 0.68em;
  font-weight: 700;
  letter-spacing: 0.04em;
}
/* Per-agent provenance chip (SPEC \xA717.3, #59): a deterministically-coloured chip on a
   multi-agent board. The hue comes from \`--dashboard-agent-hue\` (set per widget); the
   fill + text derive from it so every agent keeps one readable colour in light + dark. */
.dashboard-widget__agent {
  display: inline-flex;
  align-items: center;
  max-width: 8rem;
  padding: 1px 6px;
  border-radius: var(--bs-radius-full, 999px);
  background: hsl(var(--dashboard-agent-hue, 250) 70% 50% / 0.18);
  color: hsl(var(--dashboard-agent-hue, 250) 65% 42%);
  font-size: 0.68em;
  font-weight: 700;
  letter-spacing: 0.02em;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
@media (prefers-color-scheme: dark) {
  .dashboard-widget__agent {
    color: hsl(var(--dashboard-agent-hue, 250) 80% 72%);
  }
}
.dashboard-widget__agent--dimmed {
  opacity: 0.5;
}
/* Filter highlight: a non-matching widget recedes so the highlighted agent stands out. */
.dashboard-widget--agent-dimmed {
  opacity: 0.4;
  filter: saturate(0.7);
}
/* The per-agent filter bar (SPEC \xA717.3, #59) \u2014 a row of agent chips above the grid. */
.dashboard-agent-filter {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
  padding: 4px 2px 8px;
}
.dashboard-agent-filter__label {
  font-size: 0.72em;
  font-weight: 600;
  color: var(--bs-text-dim, #9ca3af);
  margin-inline-end: 2px;
}
.dashboard-agent-filter__chip {
  display: inline-flex;
  align-items: center;
  padding: 2px 10px;
  border-radius: var(--bs-radius-full, 999px);
  border: 1px solid var(--bs-border, #e5e7eb);
  background: transparent;
  color: var(--bs-text, #111827);
  font-size: 0.72em;
  font-weight: 600;
  cursor: pointer;
}
.dashboard-agent-filter__chip--agent {
  border-color: hsl(var(--dashboard-agent-hue, 250) 60% 55% / 0.5);
  color: hsl(var(--dashboard-agent-hue, 250) 60% 40%);
}
.dashboard-agent-filter__chip--active {
  background: hsl(var(--dashboard-agent-hue, 250) 70% 50% / 0.16);
  border-color: hsl(var(--dashboard-agent-hue, 250) 65% 50%);
}
.dashboard-agent-filter__chip--active:not(.dashboard-agent-filter__chip--agent) {
  background: color-mix(in srgb, var(--bs-accent, #6366f1) 16%, transparent);
  border-color: var(--bs-accent, #6366f1);
}
@media (prefers-color-scheme: dark) {
  .dashboard-agent-filter__chip--agent {
    color: hsl(var(--dashboard-agent-hue, 250) 80% 72%);
  }
}
/* Per-agent scope line in an approvals row (SPEC \xA717.3, #59). */
.dashboard-approvals__scope {
  font-size: 0.72em;
  color: var(--bs-text-dim, #9ca3af);
}
.dashboard-widget__handle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  color: var(--bs-text-dim, #9ca3af);
  cursor: grab;
}
.dashboard-widget__handle svg {
  width: 14px;
  height: 14px;
}
.dashboard-widget__menu {
  position: absolute;
  inset-inline-end: 6px;
  top: 34px;
  min-width: 160px;
  display: flex;
  flex-direction: column;
  padding: 4px;
  background: var(--bs-card, #fff);
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
  box-shadow: var(--bs-shadow-md, 0 8px 24px rgba(0, 0, 0, 0.18));
  z-index: 30;
}
.dashboard-widget__menu-item {
  text-align: start;
  padding: 6px 8px;
  border: none;
  border-radius: var(--bs-radius-sm, 6px);
  background: transparent;
  color: var(--bs-text, #1a1d21);
  font: inherit;
  cursor: pointer;
}
.dashboard-widget__menu-item:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}
.dashboard-widget__menu-item--danger {
  color: var(--bs-danger, #ef4444);
}
.dashboard-widget__body {
  flex: 1;
  min-height: 0;
  padding: 14px 16px;
  overflow: auto;
  border-end-start-radius: var(--bs-radius-lg, 12px);
  border-end-end-radius: var(--bs-radius-lg, 12px);
  animation: dashboard-widget-expand 160ms var(--bs-ease-out, ease-out) both;
}
@keyframes dashboard-widget-expand {
  from {
    opacity: 0;
    transform: translateY(-4px);
  }
  to {
    opacity: 1;
    transform: none;
  }
}
.dashboard-widget__resize {
  position: absolute;
  inset-inline-end: 0;
  inset-block-end: 0;
  width: 16px;
  height: 16px;
  cursor: nwse-resize;
  touch-action: none;
  background: linear-gradient(
    135deg,
    transparent 50%,
    color-mix(in srgb, var(--bs-border-strong, #d1d5db) 80%, transparent) 50%
  );
  opacity: 0;
  transition: opacity 120ms var(--bs-ease-out, ease-out);
}
.dashboard-widget:hover .dashboard-widget__resize,
.dashboard-widget:focus-within .dashboard-widget__resize,
.dashboard-widget--dragging .dashboard-widget__resize {
  opacity: 1;
}
.dashboard-widget__placeholder {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.85em;
}
.dashboard-widget__error {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  padding: 8px;
  border-radius: var(--bs-radius-sm, 6px);
  background: color-mix(in srgb, var(--bs-danger, #ef4444) 12%, transparent);
  color: var(--bs-text, #1a1d21);
}
.dashboard-widget__error-title {
  font-weight: 600;
  color: var(--bs-danger, #ef4444);
}
.dashboard-widget__error-humane {
  font-size: 0.9em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-widget__error-detail,
.dashboard-error-detail {
  font-size: 0.82em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-widget__error-detail summary,
.dashboard-error-detail summary {
  cursor: pointer;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-widget__error-message,
.dashboard-error-detail__text {
  margin-top: 4px;
  font-size: 0.85em;
  color: var(--bs-text-muted, #6b7280);
  overflow-wrap: anywhere;
}

/* --- Custom widget host -------------------------------------------------- */

.dashboard-widget__custom {
  display: flex;
  min-height: 120px;
  height: 100%;
}
.dashboard-widget__frame {
  display: block;
  flex: 1;
  width: 100%;
  min-height: 120px;
  border: 0;
  border-radius: var(--bs-radius-sm, 6px);
  background: var(--bs-card, #fff);
}
.dashboard-widget__approval {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 8px;
  padding: 8px;
}
.dashboard-widget__approval-title {
  font-weight: 600;
  color: var(--bs-text, #1a1d21);
}
.dashboard-widget__approval-sub {
  font-size: 0.85em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-widget__approval-actions {
  display: flex;
  gap: 8px;
}

/* --- Builtin bodies ------------------------------------------------------ */

.dashboard-stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dashboard-stat__value {
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.1;
  color: var(--bs-text, #1a1d21);
}
.dashboard-stat__label {
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-markdown {
  font-size: 0.9em;
}
.dashboard-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 0.85em;
}
.dashboard-list__row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 2px;
  min-width: 0;
}
.dashboard-list__row--disabled {
  opacity: 0.55;
}
.dashboard-list__link {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 0;
  color: inherit;
  text-decoration: none;
  border-radius: var(--bs-radius-sm, 6px);
  padding: 2px 4px;
}
.dashboard-list__link:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}
.dashboard-list__label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--bs-text, #1a1d21);
}
.dashboard-list__meta {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.9em;
  white-space: nowrap;
}
.dashboard-dot {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--bs-text-dim, #9ca3af);
}
.dashboard-dot--live,
.dashboard-dot--ok {
  background: var(--bs-success, #22c55e);
}
.dashboard-dot--live {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--bs-success, #22c55e) 30%, transparent);
}
.dashboard-dot--warn {
  background: var(--bs-warning, #f59e0b);
}
.dashboard-badge {
  flex: none;
  padding: 1px 6px;
  border-radius: var(--bs-radius-full, 999px);
  font-size: 0.9em;
  font-weight: 600;
  text-transform: lowercase;
}
.dashboard-badge--ok {
  background: color-mix(in srgb, var(--bs-success, #22c55e) 18%, transparent);
  color: var(--bs-success, #16a34a);
}
.dashboard-badge--error {
  background: color-mix(in srgb, var(--bs-danger, #ef4444) 16%, transparent);
  color: var(--bs-danger, #ef4444);
}
.dashboard-badge--muted {
  background: var(--bs-bg-muted, #f3f4f6);
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-usage {
  display: flex;
  gap: 20px;
}
.dashboard-usage__value {
  font-size: 1.3em;
  font-weight: 700;
  color: var(--bs-text, #1a1d21);
}
.dashboard-usage__label {
  font-size: 0.78em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-table {
  font-size: 0.82em;
}
.dashboard-table__grid {
  width: 100%;
  border-collapse: collapse;
}
.dashboard-table__grid th,
.dashboard-table__grid td {
  text-align: start;
  padding: 4px 8px;
  border-bottom: 1px solid var(--bs-border, #e5e7eb);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 220px;
}
.dashboard-table__grid th {
  color: var(--bs-text-muted, #6b7280);
  font-weight: 600;
}
.dashboard-table__footer {
  margin-top: 6px;
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.9em;
}
.dashboard-feed {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 0.84em;
}
.dashboard-feed__head {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.dashboard-feed__title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-weight: 600;
  color: var(--bs-text, #1a1d21);
}
.dashboard-feed__time {
  color: var(--bs-text-muted, #6b7280);
  white-space: nowrap;
}
.dashboard-feed__detail {
  color: var(--bs-text-muted, #6b7280);
  overflow-wrap: anywhere;
}
.dashboard-embed__frame {
  width: 100%;
  height: 100%;
  min-height: 120px;
  border: none;
  border-radius: var(--bs-radius-sm, 6px);
  background: var(--bs-bg-muted, #f3f4f6);
}

/* --- Header breadcrumb --------------------------------------------------- */

.dashboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}
.dashboard-header__breadcrumb {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.9em;
}
.dashboard-header__breadcrumb-link {
  color: var(--bs-text-muted, #6b7280);
  text-decoration: none;
}
.dashboard-header__breadcrumb-link:hover {
  color: var(--bs-text, #1a1d21);
}
.dashboard-header__breadcrumb-sep {
  color: var(--bs-text-dim, #9ca3af);
}
.dashboard-header__breadcrumb-current {
  color: var(--bs-text, #1a1d21);
  font-weight: 600;
}
.dashboard-header__breadcrumb-segment {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.dashboard-header__breadcrumb-context {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.dashboard-header__actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

/* --- Responsive: single column below ~900px ------------------------------ */

@media (max-width: 900px) {
  .dashboard-grid {
    grid-template-columns: minmax(0, 1fr);
    grid-auto-rows: auto;
  }
  .dashboard-grid > .dashboard-widget {
    grid-column: 1 / -1 !important;
    grid-row: auto !important;
    min-height: 160px;
  }
}

/* --- Empty / onboarding states ------------------------------------------ */

.dashboard-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex: 1;
  min-height: 200px;
  padding: 32px;
  text-align: center;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-empty__title {
  font-size: 1.05em;
  font-weight: 600;
  color: var(--bs-text, #1a1d21);
}
.dashboard-empty__cmd,
.dashboard-onboarding__cmd {
  margin-top: 4px;
  padding: 4px 8px;
  border-radius: var(--bs-radius-sm, 6px);
  background: var(--bs-bg-muted, #f3f4f6);
  font-family: var(--bs-font-mono, ui-monospace, monospace);
  font-size: 0.85em;
}
.dashboard-empty--tab {
  gap: 10px;
  border: 1px dashed var(--bs-border-strong, #d1d5db);
  border-radius: var(--bs-radius-lg, 12px);
  background: color-mix(in srgb, var(--bs-card, #fff) 60%, transparent);
}
.dashboard-empty__icon {
  color: var(--bs-text-dim, #9ca3af);
}
.dashboard-empty__icon svg {
  width: 28px;
  height: 28px;
}
.dashboard-onboarding {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 14px 16px;
  border: 1px solid color-mix(in srgb, var(--bs-accent, #6366f1) 40%, var(--bs-border, #e5e7eb));
  border-radius: var(--bs-radius-lg, 12px);
  background: color-mix(in srgb, var(--bs-accent, #6366f1) 8%, var(--bs-card, #fff));
}
.dashboard-onboarding__icon {
  flex: none;
  color: var(--bs-accent, #6366f1);
  margin-top: 1px;
}
.dashboard-onboarding__icon svg {
  width: 18px;
  height: 18px;
}
.dashboard-onboarding__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dashboard-onboarding__title {
  font-weight: 600;
  color: var(--bs-text-strong, #111418);
}
.dashboard-onboarding__sub {
  font-size: 0.9em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-onboarding__dismiss {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  padding: 0;
  border: none;
  border-radius: var(--bs-radius-sm, 6px);
  background: transparent;
  color: var(--bs-text-muted, #6b7280);
  cursor: pointer;
}
.dashboard-onboarding__dismiss:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
  color: var(--bs-text, #1a1d21);
}
.dashboard-onboarding__dismiss svg {
  width: 15px;
  height: 15px;
}

/* --- Skeleton loading ---------------------------------------------------- */

.dashboard-skeleton {
  display: grid;
  grid-template-columns: repeat(12, minmax(0, 1fr));
  gap: var(--dashboard-grid-gap, 12px);
  flex: 1;
  align-content: start;
}
.dashboard-skeleton__card {
  grid-column: span 4;
  min-height: 120px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-lg, 12px);
  background: linear-gradient(
    100deg,
    var(--bs-card, #fff) 30%,
    color-mix(in srgb, var(--bs-text, #1a1d21) 6%, var(--bs-card, #fff)) 50%,
    var(--bs-card, #fff) 70%
  );
  background-size: 200% 100%;
  animation: dashboard-skeleton-shimmer 1.4s ease-in-out infinite;
}
@keyframes dashboard-skeleton-shimmer {
  from {
    background-position: 200% 0;
  }
  to {
    background-position: -200% 0;
  }
}
@media (max-width: 900px) {
  .dashboard-skeleton {
    grid-template-columns: minmax(0, 1fr);
  }
  .dashboard-skeleton__card {
    grid-column: 1 / -1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .dashboard-widget__body,
  .dashboard-widget__resize,
  .dashboard-skeleton__card {
    animation: none;
    transition: none;
  }
}

/* --- chart widget (wave-charts) \u2014 dependency-free inline-SVG timeseries -- */
.dashboard-chart {
  width: 100%;
  height: 100%;
  min-height: 40px;
  display: flex;
}
.dashboard-chart__svg {
  width: 100%;
  height: 100%;
}
.dashboard-chart__line {
  stroke: var(--bs-accent, #6366f1);
  stroke-width: 1.5;
  stroke-linejoin: round;
  stroke-linecap: round;
  vector-effect: non-scaling-stroke;
}
.dashboard-chart__area {
  fill: color-mix(in srgb, var(--bs-accent, #6366f1) 15%, transparent);
  stroke: none;
}
.dashboard-chart__bars rect {
  fill: var(--bs-accent, #6366f1);
}
.dashboard-chart__gauge-track {
  stroke: var(--bs-bg-muted, #f3f4f6);
  stroke-width: 3;
  stroke-linecap: round;
  vector-effect: non-scaling-stroke;
}
.dashboard-chart__gauge-fill {
  stroke: var(--bs-accent, #6366f1);
  stroke-width: 3;
  stroke-linecap: round;
  vector-effect: non-scaling-stroke;
}
.dashboard-chart__gauge-needle {
  stroke: var(--bs-text, #1a1d21);
  stroke-width: 1;
  vector-effect: non-scaling-stroke;
}

/* --- notes widget (wave-notes) \u2014 editable write-back pad ----------------- */
.dashboard-notes {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.dashboard-notes__pad {
  flex: 1 1 auto;
  min-height: 72px;
  width: 100%;
  resize: vertical;
  padding: 8px;
  font: inherit;
  color: var(--bs-text, #1a1d21);
  background: var(--bs-input, #fff);
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-sm, 6px);
}
.dashboard-notes__hint {
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
}

/* --- action-form widget (wave-m1) \u2014 operator-authored prompt form -------- */
.dashboard-action-form {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}
.dashboard-action-form__field {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.dashboard-action-form__label {
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-action-form__control {
  padding: 6px 8px;
  font: inherit;
  color: var(--bs-text, #1a1d21);
  background: var(--bs-input, #fff);
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-sm, 6px);
}
.dashboard-action-form__submit {
  align-self: flex-start;
}

/* --- action-button widget (M5d-1) \u2014 invoke a granted external tool -------- */
.dashboard-action-button {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}
.dashboard-action-button__invoke {
  align-self: flex-start;
}
.dashboard-action-button__status,
.dashboard-action-button__hint {
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-action-button__pending {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-action-button__pending-actions {
  display: inline-flex;
  gap: 6px;
}
.dashboard-action-button__result-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-action-button__result-body {
  margin: 4px 0 0;
  max-height: 160px;
  overflow: auto;
  padding: 8px;
  font: 12px/1.5 var(--bs-font-mono, ui-monospace, monospace);
  white-space: pre-wrap;
  word-break: break-word;
  color: var(--bs-text, #1a1d21);
  background: var(--bs-surface-muted, #f3f4f6);
  border-radius: var(--bs-radius-sm, 6px);
}
.dashboard-action-button__error {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 12px;
  color: var(--bs-danger, #b91c1c);
}
.dashboard-action-button__status[data-status="confirmed"] {
  color: var(--bs-success, #15803d);
}

/* --- agent-status + approvals widgets (wave-ops) ------------------------- */
.dashboard-agent-status,
.dashboard-approvals {
  width: 100%;
}
.dashboard-approvals__actions {
  display: inline-flex;
  gap: 6px;
  margin-inline-start: auto;
}

/* --- chat widget (wave-chat) \u2014 control-plane chat face (SPEC \xA714) --------- */
.dashboard-chat {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.dashboard-chat__scroll {
  flex: 1 1 auto;
  min-height: 96px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  padding-inline-end: 2px;
}
.dashboard-chat__empty {
  margin: auto;
  padding: 12px;
  text-align: center;
  color: var(--bs-text-muted);
  font-size: 13px;
}
.dashboard-chat__turn {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dashboard-chat__role {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  text-transform: uppercase;
  color: var(--bs-text-dim);
}
.dashboard-chat__turn--user .dashboard-chat__role {
  color: var(--bs-accent);
}
.dashboard-chat__text {
  color: var(--bs-text);
  font-size: 14px;
  line-height: 1.5;
  overflow-wrap: anywhere;
}
.dashboard-chat__text pre {
  padding: 8px;
  overflow-x: auto;
  background: var(--bs-bg-muted);
  border: 1px solid var(--bs-border);
  border-radius: var(--bs-radius-sm);
  font-family: var(--bs-font-mono);
  font-size: 12px;
}
.dashboard-chat__error {
  display: flex;
  align-items: baseline;
  gap: 6px;
  padding: 6px 8px;
  border-radius: var(--bs-radius-sm);
  background: var(--bs-danger-subtle);
  color: var(--bs-danger);
  font-size: 12px;
}
.dashboard-chat__error-retry {
  color: var(--bs-text-muted);
  font-style: italic;
}
/* Tool-call group chip: a run of consecutive calls, collapsed by default. */
.dashboard-chat__tools {
  border: 1px solid var(--bs-border);
  border-radius: var(--bs-radius-md);
  background: var(--bs-card-highlight);
}
.dashboard-chat__chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 12px;
  color: var(--bs-text-muted);
  list-style: none;
}
.dashboard-chat__chip::-webkit-details-marker {
  display: none;
}
.dashboard-chat__chip-count {
  color: var(--bs-text);
}
.dashboard-chat__chip-sep {
  color: var(--bs-text-dim);
}
.dashboard-chat__chip-marks {
  font-family: var(--bs-font-mono);
  letter-spacing: 1px;
}
.dashboard-chat__tool-log {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 0 10px 8px;
}
.dashboard-chat__tool-row {
  display: flex;
  flex-direction: column;
  gap: 3px;
  font-size: 12px;
  color: var(--bs-text);
}
.dashboard-chat__tool-name {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.dashboard-chat__tool-mark {
  font-family: var(--bs-font-mono);
}
.dashboard-chat__tool-row[data-status="ok"] .dashboard-chat__tool-mark {
  color: var(--bs-success);
}
.dashboard-chat__tool-row[data-status="error"] .dashboard-chat__tool-mark {
  color: var(--bs-danger);
}
.dashboard-chat__tool-row[data-status="cancelled"] {
  opacity: 0.55;
}
.dashboard-chat__tool-detail summary {
  cursor: pointer;
  color: var(--bs-text-dim);
  font-size: 11px;
}
.dashboard-chat__tool-detail pre {
  margin: 4px 0 0;
  padding: 6px;
  overflow-x: auto;
  background: var(--bs-bg-muted);
  border-radius: var(--bs-radius-sm);
  font-family: var(--bs-font-mono);
  font-size: 11px;
}
.dashboard-chat__tool-row--building {
  flex-direction: row;
  align-items: center;
  color: var(--bs-text-muted);
}
.dashboard-chat__tool-note {
  font-style: italic;
  color: var(--bs-text-dim);
}
.dashboard-chat__shimmer {
  width: 14px;
  height: 14px;
  border-radius: var(--bs-radius-full);
  background: linear-gradient(
    90deg,
    var(--bs-bg-muted),
    var(--bs-border-strong),
    var(--bs-bg-muted)
  );
  background-size: 200% 100%;
  animation: dashboard-chat-shimmer 1.2s ease-in-out infinite;
}
@keyframes dashboard-chat-shimmer {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
@media (prefers-reduced-motion: reduce) {
  .dashboard-chat__shimmer {
    animation: none;
  }
}
/* Inline "the agent scaffolded a widget" approval card. */
.dashboard-chat__approval {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  padding: 8px 10px;
  border: 1px solid var(--bs-border-strong);
  border-radius: var(--bs-radius-md);
  background: var(--bs-card-highlight);
}
.dashboard-chat__approval-title {
  flex: 1 1 auto;
  font-size: 13px;
  color: var(--bs-text);
}
.dashboard-chat__approval-actions {
  display: inline-flex;
  gap: 6px;
}
.dashboard-chat__jump {
  align-self: center;
  padding: 3px 12px;
  border: 1px solid var(--bs-border);
  border-radius: var(--bs-radius-full);
  background: var(--bs-card);
  color: var(--bs-text-muted);
  font-size: 12px;
  cursor: pointer;
  box-shadow: var(--bs-shadow-md);
}
.dashboard-chat__input {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.dashboard-chat__textarea {
  width: 100%;
  resize: vertical;
  min-height: 42px;
  padding: 8px;
  font: inherit;
  color: var(--bs-text);
  background: var(--bs-input);
  border: 1px solid var(--bs-border);
  border-radius: var(--bs-radius-sm);
}
.dashboard-chat__input-actions {
  display: flex;
  justify-content: flex-end;
  gap: 6px;
}
.dashboard-chat__hint {
  font-size: 12px;
  color: var(--bs-text-muted);
}

/* --- ephemeral (temporary) widget badge (wave-m1) ------------------------ */
.dashboard-widget__ephemeral {
  display: inline-flex;
  align-items: center;
  padding: 1px 6px;
  border-radius: var(--bs-radius-full, 999px);
  border: 1px solid var(--bs-border, #e5e7eb);
  background: var(--bs-bg-muted, #f3f4f6);
  color: var(--bs-text-muted, #6b7280);
  font-size: 11px;
  line-height: 1.4;
}

/* --- preview widget (wave2b) \u2014 sandboxed frame + viewport presets -------- */
.dashboard-preview {
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  height: 100%;
  min-height: 0;
}
.dashboard-preview__toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  flex-wrap: wrap;
}
.dashboard-preview__viewports {
  display: inline-flex;
  gap: 4px;
}
.dashboard-preview__viewport,
.dashboard-preview__reload {
  padding: 2px 8px;
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
  background: var(--bs-bg-muted, #f3f4f6);
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-sm, 6px);
  cursor: pointer;
}
.dashboard-preview__viewport:hover,
.dashboard-preview__reload:hover {
  color: var(--bs-text, #1a1d21);
}
.dashboard-preview__frame-wrap {
  flex: 1 1 auto;
  display: flex;
  justify-content: center;
  min-height: 120px;
}
.dashboard-preview__frame-wrap--tablet .dashboard-preview__frame {
  width: 768px;
  max-width: 100%;
}
.dashboard-preview__frame-wrap--mobile .dashboard-preview__frame {
  width: 375px;
  max-width: 100%;
}

/* --- multi-operator presence + private tab + per-agent nesting (wave-w4) - */
.dashboard-tab__private {
  display: inline-flex;
  align-items: center;
  margin-inline-start: 2px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-tab__private svg {
  width: 12px;
  height: 12px;
}
.dashboard-tab__presence {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  margin-inline-start: 2px;
}
.dashboard-tab__presence-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: var(--bs-accent, #6366f1);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--bs-accent, #6366f1) 30%, transparent);
}
.dashboard-tab__presence-count {
  font-size: 11px;
  line-height: 1;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-tab-group {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 2px 4px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
}
.dashboard-tab-group__toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  height: 28px;
  padding: 0 8px;
  border: none;
  border-radius: var(--bs-radius-sm, 6px);
  background: transparent;
  color: var(--bs-text-muted, #6b7280);
  font: inherit;
  cursor: pointer;
}
.dashboard-tab-group__toggle:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
  color: var(--bs-text, #1a1d21);
}
.dashboard-tab-group__chevron svg {
  width: 14px;
  height: 14px;
}
.dashboard-tab-group__label {
  white-space: nowrap;
}
.dashboard-tab-group__count {
  min-width: 18px;
  padding: 0 5px;
  border-radius: 999px;
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
  color: var(--bs-text-muted, #6b7280);
  font-size: 12px;
  line-height: 18px;
  text-align: center;
}

/* --- full-bleed layout (wave-w3) ----------------------------------------- */
.dashboard-fullbleed {
  display: flex;
  flex-direction: column;
  min-height: 320px;
  height: 100%;
  overflow: auto;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
  background: var(--bs-card, #fff);
}
.dashboard-fullbleed > * {
  flex: 1 1 auto;
  min-height: 0;
}

/* --- widget gallery (wave-w3) -------------------------------------------- */
.dashboard-gallery {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: min(560px, 80vw);
}
.dashboard-gallery__header {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.dashboard-gallery__browse {
  display: flex;
  gap: 8px;
  align-items: center;
}
.dashboard-gallery__browse .bs-dialog__input {
  flex: 1 1 auto;
}
.dashboard-gallery__list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 40vh;
  overflow: auto;
  /* Scroll affordance (issue #4): a fade + shadow at a cut-off edge, revealed only
     when there is more list above/below. Pure CSS \u2014 the classic scroll-shadow trick:
     \`local\` cover gradients (card-colored) ride WITH the content and mask the shadow
     at whichever edge is fully scrolled to, while the \`scroll\`-attached shadows stay
     pinned to the box. No JS, no scroll listeners. */
  background:
    linear-gradient(var(--bs-card, #fff) 30%, transparent) top / 100% 22px no-repeat local,
    linear-gradient(transparent, var(--bs-card, #fff) 70%) bottom / 100% 22px no-repeat local,
    radial-gradient(
        farthest-side at 50% 0,
        color-mix(in srgb, var(--bs-text, #000) 14%, transparent),
        transparent
      )
      top / 100% 9px no-repeat scroll,
    radial-gradient(
        farthest-side at 50% 100%,
        color-mix(in srgb, var(--bs-text, #000) 14%, transparent),
        transparent
      )
      bottom / 100% 9px no-repeat scroll;
}
.dashboard-gallery__item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 8px 10px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-sm, 6px);
}
.dashboard-gallery__item-body {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.dashboard-gallery__item-name {
  font-weight: 600;
}
.dashboard-gallery__item-desc,
.dashboard-gallery__empty {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.9em;
}
.dashboard-gallery__caps {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 6px;
}
.dashboard-gallery__caps-label {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.85em;
  width: 100%;
}
.dashboard-gallery__cap {
  display: inline-flex;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--bs-border, #e5e7eb);
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
  font-family: var(--bs-font-mono, ui-monospace, monospace);
  font-size: 0.8em;
}
.dashboard-gallery__pending-note {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.85em;
}

/* --- time-travel history panel (wave-m2) --------------------------------- */
.dashboard-history {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-width: min(78vw, 820px);
  max-width: 820px;
}
.dashboard-history__header {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.dashboard-history__body {
  display: grid;
  grid-template-columns: minmax(180px, 240px) 1fr;
  gap: 16px;
  align-items: start;
}
.dashboard-history__list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin: 0;
  padding: 0;
  list-style: none;
  max-height: 60vh;
  overflow-y: auto;
}
.dashboard-history__item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
  background: transparent;
  color: var(--bs-text, #1a1d21);
  font: inherit;
  text-align: start;
  cursor: pointer;
}
.dashboard-history__item:hover {
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}
.dashboard-history__item--active {
  border-color: var(--bs-accent, #6366f1);
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}
.dashboard-history__version {
  font-weight: 600;
}
.dashboard-history__time,
.dashboard-history__latest {
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-history__latest {
  color: var(--bs-accent, #6366f1);
}
/* Per-row change summary ("+2 \xB7 1 moved \xB7 agent") under the version label. */
.dashboard-history__change {
  display: flex;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 6px;
  font-size: 12px;
}
.dashboard-history__change-label {
  color: var(--bs-text, #1a1d21);
  font-variant-numeric: tabular-nums;
}
.dashboard-history__detail {
  display: flex;
  flex-direction: column;
  gap: 14px;
  min-width: 0;
}
.dashboard-history__section-title {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--bs-text-muted, #6b7280);
  margin-bottom: 6px;
}
.dashboard-history__preview-wrap {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.dashboard-history__preview {
  position: relative;
  border: 1px dashed var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-md, 8px);
  padding: 8px;
  opacity: 0.85;
}
.dashboard-history__preview--empty {
  display: block;
  font-size: 13px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-history__cell {
  display: flex;
  align-items: center;
  gap: 6px;
  overflow: hidden;
  padding: 6px 8px;
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: var(--bs-radius-sm, 6px);
  background: var(--bs-bg-muted, #f3f4f6);
}
.dashboard-history__cell-glyph {
  flex: none;
  width: 16px;
  height: 16px;
  color: var(--bs-text-dim, #9ca3af);
}
.dashboard-history__cell-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}
/* Caption under the snapshot grid: "Layout at version N". */
.dashboard-history__preview-caption {
  margin-top: 6px;
  font-size: 11px;
  color: var(--bs-text-dim, #9ca3af);
  text-align: center;
}
.dashboard-history__diff-groups {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dashboard-history__diff-actor {
  font-size: 12px;
  font-weight: 600;
  color: var(--bs-text, #1a1d21);
  margin-bottom: 4px;
}
.dashboard-history__diff-list {
  margin: 0;
  padding: 0;
  list-style: none;
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.dashboard-history__diff-item {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 13px;
}
.dashboard-history__diff-kind {
  flex-shrink: 0;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-history__diff-label {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-history__diff-detail {
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-history__restore {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.dashboard-history__confirm {
  font-size: 13px;
}
@media (max-width: 640px) {
  .dashboard-history__body {
    grid-template-columns: 1fr;
  }
}

/* --- blame line in the cell menu (wave-m2) ------------------------------- */
.dashboard-widget__blame {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--bs-border, #e5e7eb);
  font-size: 12px;
  color: var(--bs-text-muted, #6b7280);
}
.dashboard-widget__blame-link {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--bs-accent, #6366f1);
  text-decoration: none;
}
.dashboard-widget__blame-link:hover {
  text-decoration: underline;
}
.dashboard-widget__blame-link svg {
  width: 12px;
  height: 12px;
}

/* ==========================================================================
 * Default theme polish \u2014 "Graphite".
 * Surfaces/color/type/spacing/radius/shadow only; no layout or structural
 * changes. These refine the existing selectors above into a world-class
 * default. An alternate theme layered after this file overrides them freely.
 * ========================================================================== */

.page-title {
  letter-spacing: -0.01em;
}
.card {
  padding: 14px 16px;
  border-color: var(--bs-border);
  box-shadow: var(--bs-shadow-md);
  background: var(--bs-card);
}
.card-title {
  font-size: 13px;
  font-weight: 600;
  letter-spacing: -0.005em;
  text-transform: uppercase;
  color: var(--bs-text-muted);
}

/* Widgets read as the same crisp surface as .card. */
.dashboard-widget {
  background: var(--bs-card);
  box-shadow: var(--bs-shadow-md);
}
.dashboard-widget__bar {
  background: var(--bs-card-highlight);
}
.dashboard-widget__title {
  font-weight: 600;
  color: var(--bs-text);
}

/* Tab strip: crisp segmented-control pill. */
.dashboard-tabs {
  gap: 2px;
  padding: 3px;
  border: 1px solid var(--bs-border);
  border-bottom: 1px solid var(--bs-border);
  border-radius: var(--bs-radius-md);
  background: var(--bs-bg-muted);
  width: fit-content;
}
.dashboard-tab {
  height: 28px;
  font-size: 0.86em;
  font-weight: 500;
  border-radius: calc(var(--bs-radius-md) - 3px);
  transition:
    background var(--bs-duration-fast) var(--bs-ease-out),
    color var(--bs-duration-fast) var(--bs-ease-out),
    box-shadow var(--bs-duration-fast) var(--bs-ease-out);
}
.dashboard-tab:hover {
  background: var(--bs-bg-hover);
}
.dashboard-tab--active {
  background: var(--bs-card);
  border-color: var(--bs-border-strong);
  color: var(--bs-text-strong);
  font-weight: 600;
  box-shadow:
    0 1px 1px rgba(0, 0, 0, 0.06),
    0 1px 3px rgba(0, 0, 0, 0.08);
}
.dashboard-tab:focus-visible,
.dashboard-tabs__hidden-item:focus-visible,
.bs-btn:focus-visible,
.dashboard-widget__collapse:focus-visible,
.dashboard-widget__menu-toggle:focus-visible {
  outline: none;
  box-shadow: var(--bs-focus-ring);
}

/* Toolbar buttons. */
.bs-btn {
  font-size: 0.86em;
  font-weight: 500;
  border-color: var(--bs-border);
  background: var(--bs-card);
  transition:
    background var(--bs-duration-fast) var(--bs-ease-out),
    border-color var(--bs-duration-fast) var(--bs-ease-out);
}
.bs-btn:hover {
  border-color: var(--bs-border-strong);
  background: var(--bs-bg-hover);
}
.bs-btn--primary {
  /* Re-assert the accent surface: the \`.bs-btn\` reset above reverts it to --bs-card. */
  background: var(--bs-accent);
  border-color: var(--bs-accent);
  color: var(--bs-accent-foreground);
  font-weight: 600;
  box-shadow: 0 1px 2px rgba(108, 91, 250, 0.25);
}
.bs-btn--primary:hover {
  background: color-mix(in srgb, var(--bs-accent) 88%, #000);
  border-color: color-mix(in srgb, var(--bs-accent) 88%, #000);
}

/* Chart: accent-driven line/area/gauge. */
.dashboard-chart__line {
  stroke: var(--bs-accent);
  stroke-width: 1.75;
}
.dashboard-chart__area {
  fill: color-mix(in srgb, var(--bs-accent) 18%, transparent);
}
.dashboard-chart__gauge-fill {
  stroke: var(--bs-accent);
}
.dashboard-chart__gauge-track {
  stroke: var(--bs-bg-muted);
}

/* Badges / status dots. */
.dashboard-badge {
  font-size: 0.78em;
  font-weight: 700;
  letter-spacing: 0.01em;
  padding: 2px 7px;
}
.dashboard-badge--error {
  background: var(--bs-danger-subtle);
}
.dashboard-dot--live {
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--bs-success) 25%, transparent);
}

/* ==========================================================================
 * Mixed-direction text (RTL pages).
 * With partial translations an RTL page still contains many English runs;
 * per-element \`unicode-bidi: plaintext\` lets each text run pick its own base
 * direction (first strong character), keeping English punctuation on the
 * correct side (fixes ".Your dashboard\u2026" artifacts) while Arabic/Farsi text
 * stays right-to-left. Scoped to text-bearing leaves; layout is untouched.
 * ========================================================================== */
[dir="rtl"]
  .dashboard
  :where(h1, h2, h3, h4, p, div, span, button, td, th, li, code, label, input, textarea),
[dir="rtl"]
  .bs-modal
  :where(h1, h2, h3, h4, p, div, span, button, td, th, li, code, label, input, textarea) {
  unicode-bidi: plaintext;
}
[dir="rtl"] .dashboard :where(input, textarea)::placeholder {
  unicode-bidi: plaintext;
}

/* chart-detail \u2014 sparkline delta coloring + opt-in detail mode (axes/grid/tips).
   Kept in one trailing block to minimize merge collisions with the base chart CSS
   above; all default charts are unaffected (rules key off --sparkline / --detail). */
.dashboard-chart--sparkline,
.dashboard-chart--detail {
  position: relative;
}
/* Sparkline: delta-colored line + trailing value badge. */
.dashboard-chart__spark--up .dashboard-chart__line,
.dashboard-chart__spark--up .dashboard-chart__spark-dot {
  stroke: var(--bs-success, #27853c);
  fill: var(--bs-success, #27853c);
}
.dashboard-chart__spark--down .dashboard-chart__line,
.dashboard-chart__spark--down .dashboard-chart__spark-dot {
  stroke: var(--bs-danger, #d92c25);
  fill: var(--bs-danger, #d92c25);
}
.dashboard-chart__spark--flat .dashboard-chart__spark-dot {
  fill: var(--bs-text-muted, #6b6b77);
}
.dashboard-chart__spark-value {
  position: absolute;
  top: 1px;
  inset-inline-end: 2px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
  color: var(--bs-text-muted, #6b6b77);
  pointer-events: none;
}
.dashboard-chart__spark-value--up {
  color: var(--bs-success, #27853c);
}
.dashboard-chart__spark-value--down {
  color: var(--bs-danger, #d92c25);
}
/* Detail mode: faint gridlines, corner axis labels, invisible hover-tip targets. */
.dashboard-chart__grid line {
  stroke: var(--bs-border, #e7e7ee);
  stroke-width: 0.5;
  vector-effect: non-scaling-stroke;
}
.dashboard-chart__tip {
  fill: transparent;
  stroke: none;
}
.dashboard-chart__axis {
  position: absolute;
  inset-inline-start: 3px;
  font-size: 10px;
  line-height: 1;
  color: var(--bs-text-muted, #6b6b77);
  pointer-events: none;
}
.dashboard-chart__axis--max {
  top: 1px;
}
.dashboard-chart__axis--min {
  bottom: 1px;
}

/* ===========================================================================
 * Template gallery \u2014 Templates tab (#60). Appended block: the widget/recipe tab
 * toggle and the recipe "what it needs" grant list. Reuses the existing
 * .dashboard-gallery__* tokens above; only the new recipe surfaces are styled here.
 * ======================================================================== */
.dashboard-gallery__tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--bs-border, #e5e7eb);
}
.dashboard-gallery__tab {
  appearance: none;
  background: transparent;
  border: 0;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  padding: 6px 12px;
  font: inherit;
  color: var(--bs-text-muted, #6b7280);
  cursor: pointer;
}
.dashboard-gallery__tab:hover {
  color: var(--bs-text, inherit);
}
.dashboard-gallery__tab.is-active {
  color: var(--bs-text, inherit);
  border-bottom-color: var(--bs-accent, #2563eb);
  font-weight: 600;
}
.dashboard-gallery__recipe-needs {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.8em;
  margin-top: 2px;
}
.dashboard-gallery__recipe-grants {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.dashboard-gallery__recipe-connector {
  border: 1px solid var(--bs-border, #e5e7eb);
  border-radius: 8px;
  padding: 8px 10px;
}
.dashboard-gallery__recipe-connector-name {
  font-weight: 600;
}
.dashboard-gallery__recipe-connector-reason {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.85em;
  margin-top: 2px;
}
.dashboard-gallery__recipe-tools {
  list-style: none;
  margin: 6px 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.dashboard-gallery__recipe-tool {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 6px;
  font-size: 0.85em;
}
.dashboard-gallery__recipe-tool code {
  font-family: var(--bs-font-mono, ui-monospace, monospace);
  font-size: 0.9em;
  padding: 1px 6px;
  border-radius: 6px;
  background: var(--bs-bg-hover, rgba(0, 0, 0, 0.05));
}
.dashboard-gallery__recipe-readonly {
  padding: 0 6px;
  border-radius: 999px;
  border: 1px solid var(--bs-border, #e5e7eb);
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.85em;
}
.dashboard-gallery__recipe-nogrants {
  color: var(--bs-text-muted, #6b7280);
  font-size: 0.9em;
}
`;var un=`/* Hermes DESKTOP skin \u2014 the board in the desktop app's macOS design language.
 *
 * A DIFFERENT language from the web skin (skin-web.css): where the web skin does a
 * near-transparent slab, an expanded display face with wide tracking, and sharp
 * (0-radius) buttons, the desktop app is macOS-native \u2014 soft chrome bars with a hairline
 * rule, the SF system font at normal tracking, and rounded controls. \`applyDesktopTheme\`
 * handles the token-expressible parts (card radius, single-shadow elevation, host font);
 * this sheet handles the class-level bits the tokens can't reach.
 *
 * Every rule is scoped under \`boardstate-view\` (touches only the embedded board, never
 * the host chrome) and every var() carries a fallback, so a non-desktop host degrades
 * cleanly to the bundle's own look. */

/* Title bar: soft chrome fill with a single bottom hairline \u2014 the mac "toolbar" feel,
   not the bundle's filled, contrasting drag handle. */
boardstate-view .dashboard-widget__bar {
  background: var(--ui-bg-chrome, transparent);
  border-bottom: 1px solid var(--ui-stroke-secondary, rgba(0, 0, 0, 0.08));
}

/* Widget titles use the host system font (SF on macOS) at a normal weight and NO wide
   tracking \u2014 macOS does not do letterspaced small caps. */
boardstate-view .dashboard-widget__title {
  font-family: inherit;
  font-weight: 600;
  font-size: 0.8125rem;
  letter-spacing: normal;
}

/* Rounded mac controls \u2014 the OPPOSITE of the web skin's sharp buttons. */
boardstate-view .bs-btn {
  border-radius: 0.375rem;
}
`;var bn={"--bs-bg":["--ui-surface-background","--ui-bg-editor"],"--bs-bg-hover":["--ui-row-hover-background","--ui-control-hover-background"],"--bs-bg-muted":["--ui-bg-chrome","--ui-bg-tertiary"],"--bs-surface-muted":["--ui-bg-chrome","--ui-bg-tertiary"],"--bs-card":["--ui-bg-elevated","--ui-surface-background"],"--bs-card-highlight":["--ui-row-active-background","--ui-bg-elevated"],"--bs-border":["--ui-stroke-secondary"],"--bs-border-strong":["--ui-stroke-primary"],"--bs-input":["--ui-bg-input","--ui-control-active-background"],"--bs-text":["--ui-text-primary"],"--bs-text-strong":["--ui-text-primary"],"--bs-text-muted":["--ui-text-tertiary"],"--bs-text-dim":["--ui-text-quaternary","--ui-text-tertiary"],"--bs-muted":["--ui-text-tertiary"],"--bs-accent":["--ui-accent"],"--bs-accent-foreground":["--ui-bg-elevated","--ui-surface-background"],"--bs-ring":["--ui-accent"],"--bs-danger":["--ui-red"],"--bs-success":["--ui-green"],"--bs-warning":["--ui-yellow"]};function hn(t){let e=`var(${t[t.length-1]})`;for(let r=t.length-2;r>=0;r--)e=`var(${t[r]}, ${e})`;return e}function Au(t){let e=t.match(/[\d.]+/g);if(!e||e.length<3)return 0;let r=/^\s*color\(/i.test(t)?1:255,[s,n,o]=e.slice(0,3).map(a=>{let i=Number(a)/r;return i<=.03928?i/12.92:Math.pow((i+.055)/1.055,2.4)});return .2126*s+.7152*n+.0722*o}function pn(t){return Au(t)<.4?"dark":"light"}function ue(t,e,r){return{schemaVersion:1,workspaceVersion:1,widgetsRegistry:{},prefs:{tabOrder:[t]},tabs:[{slug:t,title:e,icon:"layoutDashboard",hidden:!1,createdBy:"system",widgets:r}]}}var Pt=(t,e,r,s,n,o,a)=>({id:t,kind:"builtin:markdown",title:e,grid:{x:r,y:s,w:n,h:o},collapsed:!1,hidden:!1,props:{markdown:a}}),F=(t,e,r,s,n,o,a,i={})=>({id:t,kind:e,title:r,grid:{x:s,y:n,w:o,h:a},collapsed:!1,hidden:!1,props:i}),ku=(t,e,r,s,n,o,a,i,l,u=null)=>({id:t,kind:"builtin:action-button",title:e,grid:{x:r,y:s,w:n,h:o},collapsed:!1,hidden:!1,props:{connector:a,tool:i,label:l,args:u}}),gn=[{id:"agent-hq",name:"Agent HQ",summary:"Live operations overview \u2014 usage, sessions, connected instances, and schedules.",doc:ue("board","Agent HQ",[Pt("header","Overview",0,0,12,2,`# Agent HQ
Live operations for this Hermes agent.`),F("usage","builtin:usage","Usage",0,2,4,3),F("instances","builtin:instances","Instances",4,2,4,3),F("sessions","builtin:sessions","Sessions",8,2,4,5),F("cron","builtin:cron","Scheduled jobs",0,5,8,3)])},{id:"usage-cost",name:"Usage & Cost",summary:"Spend and token usage at a glance, with the underlying breakdown.",doc:ue("board","Usage & Cost",[Pt("header","Overview",0,0,12,2,`# Usage & Cost
Today's spend and token consumption.`),F("cost","builtin:stat-card","Cost",0,2,3,2,{metric:"todayCost",format:"usd",label:"Cost (today)"}),F("tokens","builtin:stat-card","Tokens",3,2,3,2,{metric:"todayTokens",format:"int",label:"Tokens (today)"}),F("usage","builtin:usage","Usage detail",6,2,6,3),F("cron","builtin:cron","Scheduled jobs",0,5,12,3)].map(t=>t.id==="cost"||t.id==="tokens"?{...t,bindings:{value:{source:"rpc",method:"usage.status"}}}:t))},{id:"sessions-monitor",name:"Sessions Monitor",summary:"Watch active sessions and connected instances in real time.",doc:ue("board","Sessions Monitor",[Pt("header","Overview",0,0,12,2,`# Sessions Monitor
Active sessions and connected instances.`),F("sessions","builtin:sessions","Sessions",0,2,7,5),F("instances","builtin:instances","Instances",7,2,5,3),F("usage","builtin:usage","Usage",7,5,5,2)])},{id:"office-ops",name:"Office Ops",summary:"Operate OfficeCLI \u2014 generate documents and workbooks through approved tools, artifacts on the board.",doc:ue("board","Office Ops",[Pt("header","Overview",0,0,12,2,"# Office Ops\nDrive **OfficeCLI** (`officecli mcp`) through operator-approved tools. Author the `officecli` connector in `boardstate.connectors.json`, approve its tools in the approvals panel, then act below."),ku("generate-report","Quarterly report",0,2,4,3,"officecli","officecli","Generate quarterly report .docx",{command:"create quarterly-report.docx"}),F("approvals","builtin:approvals","Approvals",4,2,8,3),Pt("setup","Setup",0,5,12,2,"### Setup\n1. Install OfficeCLI (`brew install officecli` or a GitHub release) so `officecli` is on PATH.\n2. Author `boardstate.connectors.json` in the state dir with the `officecli` stdio connector.\n3. Approve the tools you want in the approvals panel \u2014 nothing runs until you do.")])}];var Eu=new Set(["dashboard.widget.approve","dashboard.capability.approve","dashboard.action.confirm","dashboard.action.deny"]);function fn(t,e){return{request(s,n,o){return Eu.has(s)?e(s,n??{}):t.request(s,n,o)},addEventListener(s,n){return t.addEventListener(s,n)},close(){t.close()},get ready(){return t.ready},get closed(){return t.closed}}}import{jsx as _t,jsxs as Ge}from"react/jsx-runtime";var qe=!1;function Mu(){if(qe||document.querySelector("style[data-boardstate]")){qe=!0;return}let t=document.createElement("style");t.setAttribute("data-boardstate",""),t.textContent=`${cn}
${un}`,document.head.appendChild(t),qe=!0}function yn(t){let e=getComputedStyle(document.body).backgroundColor||"rgb(0,0,0)";t.setAttribute("data-theme",pn(e));for(let[r,s]of Object.entries(bn))t.style.setProperty(r,hn(s));t.style.setProperty("--bs-radius-lg","var(--radius-xl, 10px)"),t.style.setProperty("--bs-radius-md","0.375rem"),t.style.setProperty("--bs-radius-sm","0.25rem"),t.style.setProperty("--bs-shadow-md","0 1px 3px rgba(0,0,0,0.10)"),t.style.setProperty("--bs-font-sans",getComputedStyle(document.body).fontFamily)}function Cu({operatorRest:t}){let e=mn(null),r=mn(void 0),[s,n]=Ve("connecting"),[o,a]=Ve(""),[i,l]=Ve(""),u=Iu(async(h,m)=>{let w=r.current;if(w&&window.confirm(`Replace the current board with the "${h}" template?`)){l(h);try{await w.request("dashboard.workspace.replace",{doc:m,actor:"user"})}catch(v){Tu.notify?.({kind:"error",message:`Template failed: ${v instanceof Error?v.message:String(v)}`})}finally{l("")}}},[]);return Nu(()=>{Mu();let h=!1,m,w,v;return(async()=>{let g=await window.hermesDesktop?.getConnection?.().catch(()=>null);if(h)return;if(!g){n("error"),a("No desktop gateway connection.");return}if(g.authMode==="oauth"){n("error"),a("The live board needs a local gateway (OAuth remote not yet supported).");return}let $=`${g.baseUrl.replace(/^http/,"ws")}/api/plugins/boardstate/ws?token=${encodeURIComponent(g.token)}`,D=async(I,Y)=>{if(!t)throw new Error("operator endpoint unavailable");return(await t("/operator",{method:"POST",body:{method:I,params:Y}}))?.result};m=fn(Ze($),D),r.current=m,w=document.createElement("boardstate-view"),w.transport=m,w.connected=!0,w.operator=!0;try{let I=t?await t("/assets-base",{method:"GET"}):void 0;w.basePath=I?.base?`${g.baseUrl.replace(/\/+$/,"")}${I.base}`:""}catch{w.basePath=""}yn(w),v=new MutationObserver(()=>w&&yn(w)),v.observe(document.documentElement,{attributes:!0,attributeFilter:["class","style","data-theme"]}),v.observe(document.body,{attributes:!0,attributeFilter:["class","style"]}),w.style.display="block",w.style.height="100%",e.current?.appendChild(w),m.ready.then(()=>!h&&n("live")).catch(I=>{h||(n("error"),a(I instanceof Error?I.message:String(I)))})})(),()=>{h=!0,v?.disconnect(),r.current=void 0;try{m?.close()}catch{}w&&w.parentNode&&w.parentNode.removeChild(w)}},[]),Ge("div",{style:{display:"flex",flexDirection:"column",height:"100%",gap:8,padding:12},children:[Ge("div",{style:{display:"flex",alignItems:"center",flexWrap:"wrap",gap:8,fontSize:12},children:[_t("span",{style:{width:8,height:8,borderRadius:"50%",background:s==="live"?"var(--ui-green, #6aa84f)":s==="error"?"var(--ui-red, #e06c75)":"var(--ui-yellow, #d0a94f)",display:"inline-block"}}),_t("span",{style:{opacity:.8},children:s==="live"?"Board connected":s==="error"?`Board unavailable${o?`: ${o}`:""}`:"Connecting to board\u2026"}),s==="live"?Ge("span",{style:{display:"flex",alignItems:"center",flexWrap:"wrap",gap:6,marginLeft:8},children:[_t("span",{style:{opacity:.7},children:"Templates:"}),gn.map(h=>_t("button",{type:"button",title:h.summary,disabled:i!=="",onClick:()=>u(h.name,h.doc),style:{cursor:i?"default":"pointer",padding:"3px 10px",borderRadius:6,border:"1px solid var(--ui-stroke-secondary, #2a2a33)",background:i===h.id?"var(--ui-row-active-background, #23232b)":"transparent",color:"inherit",opacity:i&&i!==h.name?.5:1},children:i===h.name?"Applying\u2026":h.name},h.id))]}):null]}),_t("div",{ref:e,style:{flex:1,minHeight:0}})]})}var Fb={id:"boardstate",name:"Board",register(t){t.register({id:"board-route",area:Su,data:{path:"/board"},render:()=>_t(Cu,{operatorRest:t.rest})}),t.register({id:"board-nav",area:Ru,data:{path:"/board",label:"Board",codicon:"dashboard"}})}};export{Fb as default};
/*! Bundled license information:

@boardstate/lit/dist/browser.js:
  (**
  * @license
  * Copyright 2019 Google LLC
  * SPDX-License-Identifier: BSD-3-Clause
  *)
  (**
  * @license
  * Copyright 2017 Google LLC
  * SPDX-License-Identifier: BSD-3-Clause
  *)
  (**
  * @license
  * Copyright 2020 Google LLC
  * SPDX-License-Identifier: BSD-3-Clause
  *)
*/
