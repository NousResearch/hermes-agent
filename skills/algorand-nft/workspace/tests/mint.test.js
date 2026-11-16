import test from 'node:test';
import assert from 'node:assert/strict';
import * as sdk from 'algosdk';
import {formBytes} from '../src/core.js';
import * as mint from '../src/mint.js';
const account=sdk.generateAccount(),sender=account.addr.toString();
const params=()=>({fee:0n,minFee:1000n,firstValid:100n,lastValid:1100n,genesisID:mint.MAINNET_ID,genesisHash:mint.from64(mint.MAINNET_HASH)});
const bytes=formBytes({description:'Hermes',mime_type:'image/png'});
const url='https://arweave.net/wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4#i';
const form={name:'Hermes',unit:'HRMS01',policy:'immutable'};
const balance=()=>new sdk.modelsv2.Account({address:sender,amount:500000n,minBalance:100000n,amountWithoutPendingRewards:500000n,pendingRewards:0n,rewards:0n,round:100n,status:'Offline',totalAppsOptedIn:0,totalAssetsOptedIn:0,totalCreatedApps:0,totalCreatedAssets:0});
const build=(changes={})=>mint.buildMint({sender,bytes,url,form,params:params(),account:balance(),...changes});
function fixture(){
 const calls=[],memory=new Map();const storage={getItem:k=>memory.get(k)??null,setItem:(k,v)=>memory.set(k,v),removeItem:k=>memory.delete(k)};
 const instance={signTransaction:groups=>{calls.push('sign');return Promise.resolve([groups[0][0].txn.signTxn(account.sk)]);}};
 const context={instance,generation:1,sender,version:1,bytes:bytes.slice(),url,form:{...form}};
 const services={params:async()=>{calls.push('params');return params();},account:async()=>{calls.push('account');return balance();}};
 let time=0;return {calls,memory,storage,context,services,tick:()=>{time=120001;},create:()=>new mint.MintController({getContext:()=>context,services,storage,now:()=>time})};
}
test('confirmation verifies exact ARC69 note and asset readback without a metadata digest',async()=>{
 const f=fixture(),c=f.create();await c.prepare();await c.sign();
 const signed=c.signed.slice(),txn=sdk.decodeSignedTransaction(signed).txn;
 f.services.send=async()=>({txid:txn.txID()});await c.broadcast();
 f.services.pending=async id=>{assert.equal(id,txn.txID());return new sdk.modelsv2.PendingTransactionResponse({poolError:'',txn:sdk.decodeSignedTransaction(signed),confirmedRound:150n,assetIndex:9007199254740993n});};
 const asset=hash=>new sdk.modelsv2.Asset({index:9007199254740993n,params:new sdk.modelsv2.AssetParams({creator:sender,total:1n,decimals:0,defaultFrozen:false,name:form.name,unitName:form.unit,url,metadataHash:hash})});
 for(const hash of [undefined,new Uint8Array(),new Uint8Array(32)]){
  f.services.asset=async()=>asset(hash);assert.deepEqual(await c.confirm(),{assetId:'9007199254740993',round:'150',txid:txn.txID()});
 }
 f.services.asset=async()=>asset(new Uint8Array(32).fill(1));await assert.rejects(c.confirm(),/parameters mismatch/);
 assert.throws(()=>c.startAnother(),/verify/);
 const altered=sdk.decodeUnsignedTransaction(c.review.unsigned);altered.note=formBytes({description:'different'});
 f.services.pending=async()=>({poolError:'',txn:{txn:altered},confirmedRound:150n,assetIndex:9007199254740993n});
 await assert.rejects(c.confirm(),/transaction mismatch/);assert.ok(c.receipt);
 const reload=f.create();assert.equal(reload.receipt.status,'pending');assert.throws(()=>reload.startAnother(),/verify/);
 f.services.pending=async()=>({poolError:'',txn:{txn},confirmedRound:150n,assetIndex:9007199254740993n});f.services.asset=async()=>asset();
 await c.confirm();c.startAnother();assert.equal(c.receipt,null);assert.equal(f.memory.size,0);
});
test('legacy ARC3 recovery key survives migration and blocks mint without any automatic reads',async()=>{
 const f=fixture(),key='hermes.algorand.mainnet.mint.v0.3',raw=JSON.stringify({txid:'OLD_PUBLIC_ID',status:'confirmed'});
 f.memory.set(key,raw);const c=f.create();
 assert.notEqual(mint.RECOVERY_KEY,key);assert.match(c.recoveryError,/legacy.*ARC.?3/i);
 assert.equal(f.memory.get(key),raw);assert.deepEqual(f.calls,[]);
 await assert.rejects(c.prepare(),/legacy/i);await assert.rejects(c.sign(),/legacy/i);await assert.rejects(c.broadcast(),/legacy/i);
 assert.equal(f.memory.get(key),raw);
});
test('real SDK immutable MainNet creation contains EXACT compact ARC69 note and no metadata hash',async()=>{
 const review=await build(),tx=sdk.decodeUnsignedTransaction(review.unsigned);
 assert.equal(tx.txID(),review.txid);assert.equal(tx.genesisID,mint.MAINNET_ID);assert.deepEqual(tx.genesisHash,params().genesisHash);
 assert.equal(tx.fee,1000n);assert.equal(tx.lastValid-tx.firstValid,200n);assert.equal(tx.type,'acfg');
 assert.equal(tx.assetConfig.total,1n);assert.equal(tx.assetConfig.decimals,0);assert.equal(tx.assetConfig.defaultFrozen,false);
 assert.equal(tx.assetConfig.assetURL,url);assert.deepEqual(tx.note,bytes);
 assert.equal(sdk.msgpackRawDecodeAsMap(review.unsigned).get('apar').has('am'),false);
 for(const k of ['manager','reserve','freeze','clawback'])assert.equal(tx.assetConfig[k],undefined);
 for(const k of ['rekeyTo','group','lease','payment','assetTransfer'])assert.equal(tx[k],undefined);
});
test('Broadcast is a separate button, persists public unsigned receipt before exactly one POST, reload locks',async()=>{
 const f=fixture(),c=f.create();await c.prepare();await c.sign();let posts=0;
 f.services.send=async signed=>{posts++;assert.ok(signed instanceof Uint8Array);assert.ok(f.memory.size);throw Error('timeout');};
 const id=c.review.txid;await assert.rejects(c.broadcast(),/OUTCOME UNKNOWN/);assert.equal(posts,1);
 const saved=JSON.parse(f.memory.get(mint.RECOVERY_KEY));assert.equal(saved.txid,id);assert.ok(saved.unsigned);assert.equal('signed' in saved,false);
 c.invalidate();f.context.instance=null;assert.equal(c.receipt.txid,id);await assert.rejects(c.prepare(),/recovery/i);await assert.rejects(c.sign(),/recovery/i);await assert.rejects(c.broadcast(),/recovery/i);
 const before=f.calls.length,d=f.create();assert.equal(d.receipt.txid,id);assert.equal(f.calls.length,before);assert.equal(posts,1);
 const g=fixture(),e=g.create();await e.prepare();await e.sign();g.storage.setItem=()=>{throw Error('denied');};g.services.send=()=>{throw Error('must not send');};
 await assert.rejects(e.broadcast(),/persist/i);assert.equal(e.receipt,null);
});
test('Prepare uses only node/account reads; Sign is immediate and never broadcasts, no attestation parameters',async()=>{
 const f=fixture(),c=f.create();assert.deepEqual(f.calls,[]);
 await c.prepare();assert.deepEqual(f.calls,['params','account']);
 assert.equal(c.review.txid,sdk.decodeUnsignedTransaction(c.review.unsigned).txID());
 const signing=c.sign();assert.equal(f.calls.at(-1),'sign');await signing;
 assert.ok(c.signed);assert.equal(c.receipt,null);assert.deepEqual(f.calls,['params','account','sign']);
 f.context.form.unit='CHANGED';assert.throws(()=>c.requireReview(),/changed/);
 f.context.form.unit=form.unit;await c.prepare();f.tick();assert.throws(()=>c.requireReview(),/stale/);
});
test('creator-manager policy changes only manager while immutable remains defaultFrozen false',async()=>{
 const r=await build({form:{...form,policy:'creator'}}),a=sdk.decodeUnsignedTransaction(r.unsigned).assetConfig;
 assert.equal(a.manager.toString(),sender);for(const k of ['reserve','freeze','clawback'])assert.equal(a[k],undefined);
 assert.equal(mint.describeMint(r.unsigned).policy,'creator');
});
test('unsafe genesis, sender, policy, UTF-8 fields, note, rounds, fees and account are rejected',async()=>{
 const changes=[
  {sender:'invalid'},{params:{...params(),genesisID:'testnet-v1.0'}},{params:{...params(),genesisHash:new Uint8Array(32)}},
  {form:{...form,policy:''}},{form:{...form,name:'é'.repeat(17)}},{form:{...form,unit:'é'.repeat(5)}},{form:{...form,name:''}},
  {url:url+'x'.repeat(100)},{url:url.replace('#i','#arc3')},{bytes:new TextEncoder().encode('{"standard":"arc3"}')},{bytes:new Uint8Array(1025)},
  {params:{...params(),firstValid:0n}},{params:{...params(),lastValid:99n}},{params:{...params(),minFee:10001n}},
  {account:{...balance(),address:sdk.generateAccount().addr.toString()}},{account:{...balance(),authAddr:sdk.generateAccount().addr}},{account:{...balance(),amount:200999n}}
 ];
 for(const change of changes)await assert.rejects(build(change));
 const r=await build({account:{...balance(),amount:201000n}});assert.equal(sdk.decodeUnsignedTransaction(r.unsigned).fee,1000n);
});
test('real Ed25519 TX signature validates; tamper, foreign signer and unsupported verifier reject',async()=>{
 const r=await build(),txn=sdk.decodeUnsignedTransaction(r.unsigned),signed=txn.signTxn(account.sk);
 assert.equal((await mint.validateSigned(signed,r.unsigned)).txn.txID(),r.txid);
 const damaged=sdk.decodeSignedTransaction(signed);damaged.sig[0]^=1;
 await assert.rejects(mint.validateSigned(sdk.encodeMsgpack(damaged),r.unsigned),/Invalid transaction signature/);
 const changed=sdk.decodeUnsignedTransaction(r.unsigned);changed.note=formBytes({description:'changed'});
 await assert.rejects(mint.validateSigned(changed.signTxn(account.sk),r.unsigned),/differs/);
 await assert.rejects(mint.validateSigned(txn.signTxn(sdk.generateAccount().sk),r.unsigned),/Foreign signer/);
 await assert.rejects(mint.validateSigned(signed,r.unsigned,{}),/unavailable/);
 const multiple=new sdk.SignedTransaction({txn,msig:{v:1,thr:1,subsig:[{pk:account.addr.publicKey}]}});
 await assert.rejects(mint.validateSigned(sdk.encodeMsgpack(multiple),r.unsigned),/single ordinary/);
 await assert.rejects(mint.validateSigned(new Uint8Array(16385),r.unsigned),/one signed/);
});
test('wallet rejection and invalid payload counts never produce a signed state',async()=>{
 for(const result of [[],[null],[new Uint8Array(),new Uint8Array()]]){
  const f=fixture(),c=f.create();await c.prepare();f.context.instance.signTransaction=()=>Promise.resolve(result);
  await assert.rejects(c.sign());assert.equal(c.signed,null);assert.equal(c.receipt,null);
 }
 const f=fixture(),c=f.create();await c.prepare();f.context.instance.signTransaction=()=>Promise.reject(Error('User rejected'));
 await assert.rejects(c.sign(),/User rejected/);assert.equal(c.busy,false);assert.equal(c.signed,null);
});
test('context change during prepare or wallet signing invalidates pending results',async()=>{
 const f=fixture(),c=f.create();let finish;
 f.services.params=()=>new Promise(resolve=>{finish=resolve;});const preparing=c.prepare();f.context.version++;finish(params());
 await assert.rejects(preparing,/changed/);assert.equal(c.review,null);
 const g=fixture(),d=g.create();await d.prepare();g.context.instance.signTransaction=groups=>new Promise(resolve=>{finish=()=>resolve([groups[0][0].txn.signTxn(account.sk)]);});
 const signing=d.sign();g.context.generation++;finish();await assert.rejects(signing,/changed/);assert.equal(d.signed,null);
});
test('saved review is derived from canonical unsigned bytes and corrupt or unsafe receipts fail closed',async()=>{
 const r=await build(),raw=JSON.stringify({unsigned:mint.to64(r.unsigned),txid:r.txid,review:{name:'spoof'},status:'confirmed'});
 const restored=mint.restoreReceipt(raw);assert.equal(restored.review.name,form.name);assert.equal(restored.status,'pending');assert.equal(restored.review.note,new TextDecoder().decode(bytes));
 assert.equal('hash' in restored.review,false);
 for(const value of ['{','null','{}',JSON.stringify({unsigned:mint.to64(r.unsigned),txid:'wrong'})]){
  const f=fixture();f.memory.set(mint.RECOVERY_KEY,value);const c=f.create();assert.match(c.recoveryError,/corrupt/);await assert.rejects(c.prepare(),/recovery/i);assert.deepEqual(f.calls,[]);
 }
 for(const mutate of [t=>{t.note=new Uint8Array();},t=>{t.assetConfig.assetMetadataHash=new Uint8Array(32).fill(1);},t=>{t.assetConfig.total=2n;},t=>{t.rekeyTo=account.addr;},t=>{t.group=new Uint8Array(32).fill(1);}]){
  const txn=sdk.decodeUnsignedTransaction(r.unsigned);mutate(txn);const unsigned=sdk.encodeUnsignedTransaction(txn);
  assert.throws(()=>mint.restoreReceipt(JSON.stringify({unsigned:mint.to64(unsigned),txid:txn.txID()})));
 }
});
test('broadcast rechecks genesis, validity, balance and context before POST; storage readback must match',async()=>{
 for(const change of [
  f=>{f.services.params=async()=>({...params(),genesisID:'wrong'});},
  f=>{f.services.params=async()=>({...params(),firstValid:301n});},
  f=>{f.services.account=async()=>({...balance(),amount:100000n});},
  f=>{f.services.account=async()=>{f.context.version++;return balance();};},
  f=>{f.storage.setItem=()=>{};}
 ]){
  const f=fixture(),c=f.create();await c.prepare();await c.sign();let posts=0;f.services.send=async()=>{posts++;};change(f);
  await assert.rejects(c.broadcast());assert.equal(posts,0);assert.equal(c.receipt,null);
 }
});
test('node response txid mismatch locks outcome unknown and never retries',async()=>{
 const f=fixture(),c=f.create();await c.prepare();await c.sign();let posts=0;f.services.send=async()=>{posts++;return {txid:'wrong'};};
 await assert.rejects(c.broadcast(),/OUTCOME UNKNOWN/);await assert.rejects(c.broadcast(),/recovery/i);assert.equal(posts,1);assert.ok(c.receipt);
});
test('confirmation rejects every ASA parameter or role mismatch and failed persistence',async()=>{
 const f=fixture(),c=f.create();await c.prepare();await c.sign();const signed=c.signed.slice();f.services.send=async()=>({txid:c.review.txid});await c.broadcast();
 f.services.pending=async()=>new sdk.modelsv2.PendingTransactionResponse({poolError:'',txn:sdk.decodeSignedTransaction(signed),confirmedRound:150n,assetIndex:42n});
 const other=sdk.generateAccount().addr.toString();
 const valid={creator:sender,total:1n,decimals:0,defaultFrozen:false,name:form.name,unitName:form.unit,url};
 for(const change of [{creator:other},{total:2n},{decimals:1},{defaultFrozen:true},{name:'different'},{unitName:'OTHER'},{url:'https://example.org/other#i'},{nameB64:new TextEncoder().encode('different')},{manager:other},{reserve:other},{freeze:other},{clawback:other}]){
  f.services.asset=async()=>new sdk.modelsv2.Asset({index:42n,params:new sdk.modelsv2.AssetParams({...valid,...change})});
  await assert.rejects(c.confirm(),/parameters mismatch/);assert.equal(c.confirmedHere,false);assert.ok(c.receipt);
 }
 f.services.asset=async()=>new sdk.modelsv2.Asset({index:42n,params:new sdk.modelsv2.AssetParams(valid)});
 f.storage.setItem=()=>{throw Error('storage denied');};await assert.rejects(c.confirm(),/storage denied/);assert.equal(c.confirmedHere,false);assert.ok(c.receipt);
});
test('unconfirmed, pool error, expired confirmation or missing history keeps recovery locked',async()=>{
 const f=fixture(),c=f.create();await c.prepare();await c.sign();const signed=c.signed.slice();f.services.send=async()=>({txid:c.review.txid});await c.broadcast();
 for(const response of [{poolError:'rejected',confirmedRound:0n,assetIndex:0n},{poolError:'',confirmedRound:0n,assetIndex:0n},{poolError:'',confirmedRound:400n,assetIndex:42n,txn:sdk.decodeSignedTransaction(signed)}]){
  f.services.pending=async()=>response;await assert.rejects(c.confirm());assert.ok(c.receipt);assert.throws(()=>c.startAnother(),/verify/);
 }
 f.services.pending=async()=>{throw Error('404 history missing');};await assert.rejects(c.confirm(),/404/);await assert.rejects(c.prepare(),/recovery/i);assert.ok(c.receipt);
});

