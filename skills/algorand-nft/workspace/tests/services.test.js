import test from 'node:test';
import assert from 'node:assert/strict';
import {createServices,NODE} from '../src/services.js';
import * as mint from '../src/mint.js';
test('finite node adapter never exposes or fetches remote metadata; POST is not retried',async()=>{
 const calls=[];let posts=0;
 const request=(method,args)=>({do:async(_,options)=>{calls.push([method,args,options]);if(method==='send'){posts++;throw Error('timeout');}return method;}});
 class Client {
  constructor(...args){assert.deepEqual(args,['',NODE,'']);}
  getTransactionParams(){return request('params',[]);}
  accountInformation(a){return request('account',[a]);}
  sendRawTransaction(b){return request('send',[b]);}
  pendingTransactionInformation(id){return request('pending',[id]);}
  getAssetByID(id){return request('asset',[id]);}
 }
 const services=createServices({Client});assert.deepEqual(calls,[]);
 assert.equal('metadata' in services,false);assert.equal('verifyRemoteMetadata' in mint,false);
 await services.params();await services.account('public');await services.pending('txid');await services.asset(42n);
 await assert.rejects(services.send(new Uint8Array([1])),/timeout/);assert.equal(posts,1);
 for(const [,,options] of calls){assert.equal(options.credentials,'omit');assert.equal(options.referrerPolicy,'no-referrer');assert.equal(options.redirect,'error');assert.ok(options.signal instanceof AbortSignal);}
});
