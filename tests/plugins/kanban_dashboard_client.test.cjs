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
    fetchJSON: async url => url.includes('/boards') ? {boards:[{slug:'default'},{slug:'other'}],current:'default'} : url.includes('/config') ? {} : typeof data === 'function' ? data(url) : data,
    buildWsUrl: async (_, params) => { mints.push(params); return 'ws://fixture/' + mints.length; },
  };
  function element(attrs={}) {
    const listeners = new Map();
    return {style:{}, offsetWidth:200, classList:{add(){},remove(){}},
      addEventListener(k, fn) { if (!listeners.has(k)) listeners.set(k,new Set()); listeners.get(k).add(fn); },
      removeEventListener(k, fn) { listeners.get(k)?.delete(fn); },
      dispatchEvent(e) { for (const fn of listeners.get(e.type) || []) fn(e); },
      getAttribute:k=>attrs[k], hasAttribute:k=>k in attrs,
      closest(selector) { return selector==='[data-kanban-column]' && attrs['data-kanban-column'] ? this : null; },
      cloneNode:()=>element(), remove(){},
    };
  }
  let hit = null;
  const document = Object.assign(element(), {body:{appendChild(){}}, elementFromPoint:()=>hit});
  const window = {__HERMES_PLUGIN_SDK__: SDK, __HERMES_PLUGINS__: {register: (_, page) => { Page = page; }}, localStorage: {getItem: k => storage.get(k), setItem: (k,v) => storage.set(k,v)}, addEventListener() {}, removeEventListener() {}};
  const context = vm.createContext({window, console, URLSearchParams, Set, WebSocket: class {constructor(url) {this.url=url; sockets.push(this);} close() {this.onclose?.({code:1000});}}, setTimeout: (fn, ms) => {const id=nextTimer++; timers.set(id,{fn,ms}); return id;}, clearTimeout: id => timers.delete(id), document, CustomEvent: class {constructor(type, init) {this.type=type; Object.assign(this,init);}}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../../plugins/kanban/dashboard/dist/index.js'),'utf8'),context);
  function mount(component, props={}) { const a={slots:[],pending:[],i:0}; return {render(next=props) {props=next; active=a; a.i=0; const tree=component(props); for (const n of nodes(tree)) { if (n.props.ref && typeof n.type === 'string') n.props.ref.current ||= element(n.props); } a.pending.splice(0).forEach(f=>f()); return tree;}, unmount() {a.slots.forEach(s=>s?.cleanup?.());}}; }
  const nodes = tree => !tree || typeof tree !== 'object' ? [] : [tree,...(tree.props?.children || []).flatMap(nodes)];
  const flush = async () => { for(let i=0;i<10;i++) await Promise.resolve(); };
  return {mount, Page, nodes, flush, document, setHit: el=>{hit=el;}, timers, sockets, mints, storage, setData: d => {data=d;}, runTimer(ms) {const entry=[...timers].find(([,t])=>t.ms===ms); assert.ok(entry,`timer ${ms} scheduled`); timers.delete(entry[0]); entry[1].fn();}};
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

test('stock columns remain present when empty, populated or filtered', async () => {
  const x=harness(), page=x.mount(x.Page);
  page.render(); await x.flush();
  const boardNode=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  assert.ok(boardNode);
  const board=x.mount(boardNode.type,boardNode.props);
  const columns=tree=>x.nodes(tree).filter(n=>n.type?.name==='Column');
  assert.deepEqual(columns(board.render()).map(n=>n.props.column.name),['ready','running','blocked','done']);
  for (const node of columns(board.render())) {
    const tree=x.mount(node.type,node.props).render();
    assert.equal(x.nodes(tree).find(n=>n.props?.className==='hermes-kanban-column-count').props.children[0],0);
  }
  const populated={...boardNode.props,board:{...boardNode.props.board,columns:boardNode.props.board.columns.map(c=>c.name==='ready'?{...c,tasks:[{id:'new'}]}:c)}};
  assert.deepEqual(columns(board.render(populated)).map(n=>n.props.column.name),['ready','running','blocked','done']);
  const filtered={...populated,board:{...populated.board,columns:populated.board.columns.map(c=>({...c,tasks:[]}))}};
  assert.deepEqual(columns(board.render(filtered)).map(n=>n.props.column.name),['ready','running','blocked','done']);
});

test('Done renders its cards without a persistent collapse control', async () => {
  const x=harness(), page=x.mount(x.Page);
  x.setData({columns:[{name:'done',tasks:[{id:'done1',status:'done'}]}],tenants:[],assignees:[]});
  page.render(); await x.flush();
  const node=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  const board=x.mount(node.type,node.props);
  let tree=board.render();
  const toggle=()=>x.nodes(tree).find(n=>n.type==='button' && n.props['aria-expanded'] !== undefined);
  assert.equal(toggle(),undefined);
  assert.equal(x.nodes(tree).filter(n=>n.type?.name==='Column').length,1);
  const column=x.nodes(tree).find(n=>n.type?.name==='Column');
  assert.equal(x.nodes(x.mount(column.type,column.props).render()).filter(n=>n.type?.name==='TaskCard').length,1);
  assert.equal(x.storage.size,0);
  board.unmount();
  const remount=x.mount(node.type,node.props);
  assert.equal(x.nodes(remount.render()).filter(n=>n.type?.name==='Column').length,1,'expanded preference survives remount');
});

test('stock card face does not expose block reasons or worker failure', async () => {
  const x=harness(), page=x.mount(x.Page);
  const reason='<img src=x onerror=alert(1)> '+ 'long '.repeat(30);
  x.setData({columns:[{name:'blocked',tasks:[{id:'blocked1',status:'blocked',block_kind:'capability',block_reason:reason,last_failure_error:reason}]}],tenants:[],assignees:[]});
  page.render(); await x.flush();
  const bn=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  const cn=x.nodes(x.mount(bn.type,bn.props).render()).find(n=>n.type?.name==='Column');
  const card=x.nodes(x.mount(cn.type,cn.props).render()).find(n=>n.type?.name==='TaskCard');
  const nodes=x.nodes(x.mount(card.type,card.props).render());
  assert.equal(nodes.some(n=>n.props.children.includes('⛔ capability')),false);
  const details=nodes.filter(n=>n.props.title===reason);
  assert.equal(details.length,0);
});

test('stock destinations support desktop and touch single/bulk moves and cleanup', async () => {
  for (const touch of [false,true]) for (const bulk of [false,true]) for (const destination of ['ready','done']) {
    const x=harness(), page=x.mount(x.Page);
    x.setData({columns:[{name:'ready',tasks:[]},{name:'running',tasks:[{id:'r1',status:'running'},{id:'r2',status:'running'}]},{name:'blocked',tasks:[]},{name:'done',tasks:[{id:'d1',status:'done'}]}],tenants:[],assignees:[]});
    page.render(); await x.flush();
    const bn=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
    const positions={r1:'running',r2:'running'};
    const props={...bn.props,selectedIds:new Set(bulk?['r1','r2']:[]),
      onMove:(id,status)=>{positions[id]=status;},
      onMoveSelected:status=>{for(const id of props.selectedIds) positions[id]=status;}};
    const board=x.mount(bn.type,props);
    let tree=board.render();
    if (touch) {
      const running=x.nodes(tree).find(n=>n.type?.name==='Column' && n.props.column.name==='running');
      const cn=x.mount(running.type,running.props).render();
      const card=x.nodes(cn).find(n=>n.type?.name==='TaskCard');
      x.mount(card.type,card.props).render().props.ref.current.dispatchEvent({type:'pointerdown',pointerType:'touch',preventDefault(){},clientX:10,clientY:10});
    } else tree.props.onDragStart({target:{closest:()=>({getAttribute:()=> 'r1'})}});
    const fresh=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
    assert.equal(fresh.props.draggingTaskId,'r1','actual drag entry updates page state');
    tree=board.render({...props,draggingTaskId:fresh.props.draggingTaskId});
    const target=x.nodes(tree).find(n=>n.type?.name==='Column' && n.props.column.name===destination);
    assert.ok(target,`${destination} is a real Column during drag`);
    const dom=x.mount(target.type,target.props).render();
    assert.equal(dom.props['data-kanban-column'],destination);
    assert.equal(typeof dom.props.onDragOver,'function');
    assert.equal(typeof dom.props.onDrop,'function');
    if (touch) {
      x.setHit(dom.props.ref.current);
      x.document.dispatchEvent({type:'pointermove',clientX:30,clientY:30});
      x.document.dispatchEvent({type:'pointerup'});
    } else {
      const e={preventDefault(){},dataTransfer:{getData:()=> 'r1'}};
      dom.props.onDragOver(e); assert.equal(e.dataTransfer.dropEffect,'move');
      dom.props.onDrop(e); tree.props.onDragEnd();
    }
    assert.equal(positions.r1,destination);
    assert.equal(positions.r2,bulk?destination:'running');
    const ended=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
    assert.equal(ended.props.draggingTaskId,null);
    tree=board.render({...props,draggingTaskId:ended.props.draggingTaskId});
    assert.equal(x.nodes(tree).filter(n=>n.type?.name==='Column' && ['ready','done'].includes(n.props.column.name)).length,2);
    assert.equal(x.storage.get('hermes.kanban.doneExpanded'),undefined,'drag does not persist expansion');
    page.unmount(); board.unmount();
  }
});

test('in-flight snapshot cannot roll back WS cursor before an auth remint', async () => {
  const x=harness(), page=x.mount(x.Page);
  page.render(); await x.flush(); page.render(); await x.flush(); page.render(); await x.flush();
  const ws=x.sockets.at(-1); ws.onopen();
  ws.onmessage({data:JSON.stringify({events:[{task_id:'r1'}],cursor:8})});
  x.setData({columns:[],tenants:[],assignees:[],latest_event_id:8});
  x.runTimer(250);
  ws.onmessage({data:JSON.stringify({events:[{task_id:'r1'}],cursor:9})});
  await x.flush();
  ws.onclose({code:1008}); x.runTimer(1000); await x.flush();
  assert.equal(x.mints.at(-1).since,'9');
  const retry=x.sockets.at(-1); retry.onopen();
  retry.onmessage({data:JSON.stringify({events:[{task_id:'r1'}],cursor:8})});
  retry.onclose({code:1008}); x.runTimer(1000); await x.flush();
  assert.equal(x.mints.at(-1).since,'9','WS messages cannot lower the cursor either');
  page.unmount();
});

test('switching boards resets to its own snapshot and discards old-board HTTP responses', async () => {
  const x=harness(), page=x.mount(x.Page);
  page.render(); await x.flush(); page.render(); await x.flush(); page.render(); await x.flush();
  const ws=x.sockets.at(-1);
  let resolveOld;
  x.setData(()=>new Promise(resolve=>{resolveOld=resolve;}));
  ws.onmessage({data:JSON.stringify({events:[{task_id:'r1'}],cursor:90})});
  x.runTimer(250);
  const switcher=x.nodes(page.render()).find(n=>n.type?.name==='BoardSwitcher');
  assert.ok(switcher);
  const snapshot={columns:[],tenants:[],assignees:[],latest_event_id:2};
  x.setData(snapshot);
  switcher.props.onSwitch('other');
  page.render(); await x.flush(); page.render(); await x.flush();
  assert.equal(x.mints.at(-1).board,'other');
  assert.equal(x.mints.at(-1).since,'2','never max across boards');
  resolveOld({...snapshot,latest_event_id:100,columns:[{name:'ready',tasks:[{id:'old-board'}]}]});
  await x.flush();
  const bn=x.nodes(page.render()).find(n=>n.type?.name==='BoardColumns');
  assert.equal(bn.props.board.columns.length,0,'stale snapshot must not replace the selected grid');
  const other=x.sockets.at(-1); other.onclose({code:1008}); x.runTimer(1000); await x.flush();
  assert.equal(x.mints.at(-1).since,'2','stale response cannot contaminate new cursor');
  page.unmount();
});

module.exports = {harness};
