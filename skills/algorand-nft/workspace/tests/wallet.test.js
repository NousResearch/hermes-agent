import test from 'node:test';
import assert from 'node:assert/strict';
import * as wallet from '../src/wallet.js';
function fixture(reject=false) {
  const calls=[], events=new Map(); let opts;
  class Fake {
    constructor(options) { opts=options; calls.push('construct'); }
    on(name, fn) { events.set(name,fn); return ()=>events.delete(name); }
    connect() { calls.push('connect'); return reject?Promise.reject({data:{type:'CONNECT_NETWORK_MISMATCH'}}):Promise.resolve(['public-one','public-two']); }
    disconnect() { calls.push('disconnect'); return Promise.resolve(); }
    dispose() { calls.push('dispose'); }
    reconnectSession() { throw Error('Forbidden'); }
    signTransaction() { throw Error('Forbidden'); }
  }
  const controller = new wallet.WalletController(Fake,a=>a.startsWith('public-'));
  return {controller,calls,events,options:()=>opts};
}
test('wallet constructed only on explicit connect with numeric MainNet; select and disconnect reset', async()=>{
  const f=fixture(); assert.deepEqual(f.calls,[]);
  const connect=f.controller.connect(); assert.deepEqual(f.calls,['construct','connect']);
  assert.equal(f.options().chainId,416001); await connect;
  assert.equal(f.controller.selected,'public-one');
  f.controller.select('public-two'); assert.equal(f.controller.selected,'public-two');
  assert.throws(()=>f.controller.select('bad'));
  f.events.get('disconnect')(); assert.equal(f.controller.selected,''); assert.equal(f.events.size,0);
  await f.controller.connect(); const start=f.calls.length; await f.controller.disconnect();
  assert.deepEqual(f.calls.slice(start),['disconnect','dispose']);
  assert.equal(f.controller.selected,''); assert.equal(f.events.size,0);
});
test('wallet rejection never reports success; network change clears session', async()=>{
  const f=fixture(true); await f.controller.connect();
  assert.equal(f.controller.selected,''); assert.match(f.controller.error,/CONNECT_NETWORK_MISMATCH/); assert.equal(f.events.size,0);
  const g=fixture(); await g.controller.connect(); g.events.get('networkChanged')({network:'testnet'});
  assert.equal(g.controller.selected,''); assert.match(g.controller.error,/network/i);
});
