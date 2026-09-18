// Run: node --test tests/plugins/kanban_dashboard_client.test.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');

function harness() {
  let Page, active, nextTimer = 1;
  const timers = new Map(), sockets = [], mints = [], storage = new Map();
  let data = { columns: ['ready', 'running', 'blocked', 'done'].map(name => ({name, tasks: []})), tenants: [], assignees: [], latest_event_id: 7 };
  const same = (a, b) => a && b && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  const hooks = {
    useState(initial) { const a = active, i = a.i++; if (!(i in a.slots)) a.slots[i] = typeof initial === 'function' ? initial() : initial; return [a.slots[i], v => { a.slots[i] = typeof v === 'function' ? v(a.slots[i]) : v; }]; },
    useRef(initial) { const a = active, i = a.i++; return a.slots[i] ||= {current: initial}; },
    useMemo(fn, deps) { const a = active, i = a.i++; if (!a.slots[i] || !same(a.slots[i].deps, deps)) a.slots[i] = {deps, value: fn()}; return a.slots[i].value; },
    useCallback(fn, deps) { return hooks.useMemo(() => fn, deps); },
    useEffect(fn, deps) { const a = active, i = a.i++; if (!a.slots[i] || !same(a.slots[i].deps, deps)) { const prev = a.slots[i]; a.slots[i] = {deps}; a.pending.push(() => { prev?.cleanup?.(); a.slots[i].cleanup = fn(); }); } },
  };
  const h = (type, props, ...children) => ({type, props: {...props, children: children.flat(Infinity)}});
  const SDK = { React: {createElement: h, Component: class {}, ...hooks}, hooks,
    components: Object.fromEntries(['Card','CardContent','Badge','Button','Input','Label','Select','SelectOption','Checkbox'].map(x => [x,x])),
    utils: {cn: (...xs) => xs.filter(Boolean).join(' '), timeAgo: () => ''},
    fetchJSON: async url => url.includes('/boards') ? {boards:[{slug:'default'}],current:'default'} : url.includes('/config') ? {} : data,
    buildWsUrl: async (_, params) => { mints.push(params); return 'ws://fixture/' + mints.length; },
  };
  const window = {__HERMES_PLUGIN_SDK__: SDK, __HERMES_PLUGINS__: {register: (_, page) => { Page = page; }}, localStorage: {getItem: k => storage.get(k), setItem: (k,v) => storage.set(k,v)}, addEventListener() {}, removeEventListener() {}};
  const context = vm.createContext({window, console, URLSearchParams, Set, WebSocket: class {constructor(url) {this.url=url; sockets.push(this);} close() {this.onclose?.({code:1000});}}, setTimeout: (fn, ms) => {const id=nextTimer++; timers.set(id,{fn,ms}); return id;}, clearTimeout: id => timers.delete(id), document: {addEventListener(){},removeEventListener(){}}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../../plugins/kanban/dashboard/dist/index.js'),'utf8'),context);
  function mount(component, props={}) { const a={slots:[],pending:[],i:0}; return {render(next=props) {props=next; active=a; a.i=0; const tree=component(props); a.pending.splice(0).forEach(f=>f()); return tree;}, unmount() {a.slots.forEach(s=>s?.cleanup?.());}}; }
  const nodes = tree => !tree || typeof tree !== 'object' ? [] : [tree,...(tree.props?.children || []).flatMap(nodes)];
  const flush = async () => { for(let i=0;i<10;i++) await Promise.resolve(); };
  return {mount, Page, nodes, flush, timers, sockets, mints, storage, setData: d => {data=d;}, runTimer(ms) {const entry=[...timers].find(([,t])=>t.ms===ms); assert.ok(entry,`timer ${ms} scheduled`); timers.delete(entry[0]); entry[1].fn();}};
}

test('auth expiry retries with a fresh ticket without reload churn; cleanup cancels retries', async () => {
  const x=harness(), page=x.mount(x.Page);
  page.render(); await x.flush(); page.render(); await x.flush(); page.render(); await x.flush();
  const ws=x.sockets.at(-1), before=x.mints.length;
  ws.onopen(); ws.onclose({code:1008});
  x.runTimer(1000); await x.flush();
  assert.equal(x.mints.length,before+1);
  const retry=x.sockets.at(-1); retry.onopen();
  retry.onmessage({data:JSON.stringify({events:[{task_id:'t_fixture'}],cursor:8})});
  x.runTimer(250); await x.flush(); page.render(); await x.flush();
  assert.equal(x.mints.length,before+1,'board reload must not reconnect');
  retry.onclose({code:1008}); page.unmount();
  assert.equal(x.timers.size,0,'no orphan retry after unmount');
});

module.exports = {harness};
