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

test('empty columns hide, watched columns remain and new tasks restore columns', async () => {
  const x=harness(), page=x.mount(x.Page);
  page.render(); await x.flush();
  const boardNode=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  assert.ok(boardNode);
  const board=x.mount(boardNode.type,boardNode.props);
  const columns=tree=>x.nodes(tree).filter(n=>n.type?.name==='Column');
  assert.deepEqual(columns(board.render()).map(n=>n.props.column.name),['running','blocked']);
  for (const node of columns(board.render())) {
    const tree=x.mount(node.type,node.props).render();
    assert.equal(x.nodes(tree).find(n=>n.props?.className==='hermes-kanban-column-count').props.children[0],'—');
  }
  const populated={...boardNode.props,board:{...boardNode.props.board,columns:boardNode.props.board.columns.map(c=>c.name==='ready'?{...c,tasks:[{id:'new'}]}:c)}};
  assert.deepEqual(columns(board.render(populated)).map(n=>n.props.column.name),['ready','running','blocked']);
});

test('Done defaults to a count stub, expands and persists the toggle', async () => {
  const x=harness(), page=x.mount(x.Page);
  x.setData({columns:[{name:'done',tasks:[{id:'done1',status:'done'}]}],tenants:[],assignees:[]});
  page.render(); await x.flush();
  const node=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  const board=x.mount(node.type,node.props);
  let tree=board.render();
  const toggle=()=>x.nodes(tree).find(n=>n.type==='button' && n.props['aria-expanded'] !== undefined);
  assert.ok(toggle(),'Done has an accessible toggle');
  assert.equal(toggle().props['aria-expanded'],false);
  assert.match(toggle().props.children.join(''),/Done · 1/);
  assert.equal(x.nodes(tree).filter(n=>n.type?.name==='Column').length,0);
  toggle().props.onClick(); tree=board.render();
  assert.equal(x.nodes(tree).filter(n=>n.type?.name==='Column').length,1);
  board.unmount();
  const remount=x.mount(node.type,node.props);
  assert.equal(x.nodes(remount.render()).filter(n=>n.type?.name==='Column').length,1,'expanded preference survives remount');
});

test('blocked cards expose kind and truncated reason/failure as plain text with full titles', async () => {
  const x=harness(), page=x.mount(x.Page);
  const reason='<img src=x onerror=alert(1)> '+ 'long '.repeat(30);
  x.setData({columns:[{name:'blocked',tasks:[{id:'blocked1',status:'blocked',block_kind:'capability',block_reason:reason,last_failure_error:reason}]}],tenants:[],assignees:[]});
  page.render(); await x.flush();
  const bn=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  const cn=x.nodes(x.mount(bn.type,bn.props).render()).find(n=>n.type?.name==='Column');
  const card=x.nodes(x.mount(cn.type,cn.props).render()).find(n=>n.type?.name==='TaskCard');
  const nodes=x.nodes(x.mount(card.type,card.props).render());
  assert.ok(nodes.some(n=>n.props.children.includes('⛔ capability')));
  const details=nodes.filter(n=>n.props.title===reason);
  assert.equal(details.length,2);
  details.forEach(n=>{assert.equal(n.props.children[0],reason.slice(0,90)+'…'); assert.equal(n.props.dangerouslySetInnerHTML,undefined);});
});

module.exports = {harness};
