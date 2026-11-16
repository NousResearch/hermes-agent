import * as sdk from 'algosdk';
import {validateNote,validateMediaURL} from './core.js';
export const MAINNET_ID='mainnet-v1.0';
export const MAINNET_HASH='wGHE2Pwdvd7S12BL5FaOP20EGYesN73ktiC1qzkkit8=';
export const from64=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
export const to64=b=>btoa(String.fromCharCode(...b));
export const equal=(a,b)=>a instanceof Uint8Array && b instanceof Uint8Array && a.length===b.length && a.every((x,i)=>x===b[i]);
export function assertMainnet(params) {
 if(params.genesisID!==MAINNET_ID || !equal(params.genesisHash,from64(MAINNET_HASH)))throw Error('Pinned MainNet genesis mismatch.');
}
export function assertAccount(account,sender,fee) {
 if(account.address!==sender)throw Error('Returned account differs from sender.');
 if(account.authAddr && account.authAddr.toString()!==sender)throw Error('Rekeyed accounts are unsupported.');
 if(typeof account.amount!=='bigint' || typeof account.minBalance!=='bigint' || account.amount-account.minBalance<fee+100000n)throw Error('Insufficient spendable balance for fee + 100000 microALGO asset minimum balance.');
}
const length=(text,min,max,label)=>{const n=new TextEncoder().encode(text).length;if(typeof text!=='string'||n<min||n>max)throw Error(`${label} must be ${min}..${max} UTF-8 bytes.`);};
export async function buildMint({sender,bytes,url,form,params,account}) {
 if(!sdk.isValidAddress(sender))throw Error('Invalid sender address.');
 assertMainnet(params);
 if(!['immutable','creator'].includes(form.policy))throw Error('Choose an administrative policy explicitly.');
 length(form.name,1,32,'Asset name');length(form.unit,1,8,'Unit');length(url,1,96,'URL');
 validateMediaURL(url);validateNote(bytes);
 const first=BigInt(params.firstValid),last=BigInt(params.lastValid);
 if(first<=0n||last<first)throw Error('Invalid validity rounds.');
 const txn=sdk.makeAssetCreateTxnWithSuggestedParamsFromObject({sender,total:1n,decimals:0,defaultFrozen:false,manager:form.policy==='creator'?sender:undefined,assetName:form.name,unitName:form.unit,assetURL:url,note:bytes,suggestedParams:{...params,flatFee:false,minFee:BigInt(params.minFee)<1000n?1000n:params.minFee,lastValid:last<first+200n?last:first+200n}});
 if(txn.fee<1000n||txn.fee>10000n)throw Error('Calculated SDK fee outside 1000..10000 microALGO cap.');
 assertAccount(account,sender,txn.fee);
 return {unsigned:sdk.encodeUnsignedTransaction(txn),txid:txn.txID(),mbr:'100000'};
}

export async function validateSigned(signed,unsigned,subtle=globalThis.crypto?.subtle) {
 if(!(signed instanceof Uint8Array)||signed.length>16384)throw Error('Expected one signed payload.');
 const decoded=sdk.decodeSignedTransaction(signed),expected=sdk.decodeUnsignedTransaction(unsigned);
 if(decoded.msig||decoded.lsig||decoded.pqsig||!decoded.sig||decoded.sig.length!==64)throw Error('Only a single ordinary 64-byte Ed25519 signature is supported.');
 if(decoded.sgnr && decoded.sgnr.toString()!==expected.sender.toString())throw Error('Foreign signer unsupported.');
 if(!equal(sdk.encodeUnsignedTransaction(decoded.txn),unsigned)||decoded.txn.txID()!==expected.txID())throw Error('Signed transaction differs from review.');
 let valid;
 try {
  const key=await subtle.importKey('raw',expected.sender.publicKey,{name:'Ed25519'},false,['verify']);
  valid=await subtle.verify({name:'Ed25519'},key,decoded.sig,decoded.txn.bytesToSign());
 } catch {throw Error('Browser Ed25519 transaction verification unavailable; signing is blocked.');}
 if(!valid)throw Error('Invalid transaction signature.');
 return decoded;
}
const sameContext=(a,b)=>a.instance===b.instance && a.generation===b.generation && a.sender===b.sender && a.version===b.version && a.url===b.url && equal(a.bytes,b.bytes) && ['name','unit','policy'].every(k=>a.form[k]===b.form[k]);
export const absentHash=value=>value===undefined||value===null||(value instanceof Uint8Array&&(value.length===0||value.length===32)&&value.every(b=>b===0));
export const RECOVERY_KEY='hermes.algorand.mainnet.mint.v0.4';
export const LEGACY_RECOVERY_KEY='hermes.algorand.mainnet.mint.v0.3';
export function describeMint(unsigned) {
 const txn=sdk.decodeUnsignedTransaction(unsigned),a=txn.assetConfig;
 assertMainnet(txn);
 if(!equal(sdk.encodeUnsignedTransaction(txn),unsigned)||txn.type!=='acfg'||!a||a.assetIndex!==0n||a.total!==1n||a.decimals!==0||a.defaultFrozen!==false||txn.group||txn.rekeyTo||txn.lease||txn.payment||txn.assetTransfer||txn.applicationCall)throw Error('Unsafe or unsupported mint transaction.');
 if(txn.fee<1000n||txn.fee>10000n)throw Error('Unsafe transaction fee.');
 if(txn.firstValid<=0n||txn.lastValid<txn.firstValid||txn.lastValid-txn.firstValid>1000n)throw Error('Unsafe validity rounds.');
 const sender=txn.sender.toString();
 if(a.reserve||a.freeze||a.clawback||(a.manager&&a.manager.toString()!==sender))throw Error('Unsafe administrative roles.');
 length(a.assetName,1,32,'Asset name');length(a.unitName,1,8,'Unit');length(a.assetURL,1,96,'URL');
 validateMediaURL(a.assetURL);validateNote(txn.note);
 if(!absentHash(a.assetMetadataHash))throw Error('Unsafe metadata hash: ARC69 must omit it.');
 return {network:MAINNET_ID,genesisHash:MAINNET_HASH,sender,txid:txn.txID(),fee:txn.fee.toString(),mbr:'100000',firstValid:txn.firstValid.toString(),lastValid:txn.lastValid.toString(),name:a.assetName,unit:a.unitName,url:a.assetURL,total:'1',decimals:0,defaultFrozen:false,manager:a.manager?.toString()||null,reserve:null,freeze:null,clawback:null,policy:a.manager?'creator':'immutable',rekey:null,close:null,group:null,lease:null,note:new TextDecoder().decode(txn.note),unsigned:to64(unsigned)};
}
export function restoreReceipt(raw) {
 const value=JSON.parse(raw);
 if(typeof value?.unsigned!=='string'||value.unsigned.length>22000)throw Error('Recovery unsigned bytes missing or oversized.');
 const unsigned=from64(value.unsigned),review=describeMint(unsigned);
 if(review.txid!==value.txid)throw Error('Recovery transaction ID mismatch.');
 return {version:4,network:MAINNET_ID,genesisHash:MAINNET_HASH,txid:review.txid,unsigned:to64(unsigned),review,status:'pending'};
}
export class MintController {
 constructor({getContext,services,storage,now=()=>Date.now(),changed=()=>{}}) {
  Object.assign(this,{getContext,services,storage,now,changed});this.busy=false;this.epoch=0;this.review=null;this.signed=null;this.receipt=null;this.recoveryError='';this.restore();
 }
 restore() {
  try {
   if(this.storage.getItem(LEGACY_RECOVERY_KEY)!==null)throw Error('Legacy ARC-3 recovery receipt detected. Preserve its public transaction ID and original storage; use the original workflow for manual recovery. Never mint a replacement.');
   const raw=this.storage.getItem(RECOVERY_KEY);if(raw!==null)this.receipt=restoreReceipt(raw);
  }
  catch(e){this.recoveryError=`Recovery storage blocked or corrupt: ${e.message}. Do not mint again; recover manually.`;}
 }
 assertNoRecovery() {
  this.restore();if(this.recoveryError||this.receipt)throw Error(this.recoveryError||'Existing recovery receipt locks new mint; check confirmation for its exact transaction ID.');
 }
 async broadcast() {
  this.assertNoRecovery();
  if(this.busy)throw Error('An operation is pending.');
  const review=this.requireReview(),signed=this.signed?.slice();
  if(!signed)throw Error('Sign the reviewed transaction first.');
  this.busy=true;this.changed();
  try {
   const params=await this.services.params();assertMainnet(params);this.requireReview();
   const txn=sdk.decodeUnsignedTransaction(review.unsigned);
   if(BigInt(params.firstValid)<txn.firstValid||BigInt(params.firstValid)>txn.lastValid)throw Error('Reviewed validity rounds expired or not current.');
   const account=await this.services.account(review.context.sender);this.requireReview();assertAccount(account,review.context.sender,txn.fee);
   await validateSigned(signed,review.unsigned);this.requireReview();this.assertNoRecovery();
   const receipt=restoreReceipt(JSON.stringify({txid:review.txid,unsigned:to64(review.unsigned)}));
   const raw=JSON.stringify(receipt);
   try {this.storage.setItem(RECOVERY_KEY,raw);if(this.storage.getItem(RECOVERY_KEY)!==raw)throw Error('Recovery readback differs.');}
   catch(e){throw Error(`Cannot persist recovery receipt; no POST attempted: ${e.message}`);}
   // Permanent in-memory lock BEFORE invoking POST; edits/disconnect never clear it.
   this.receipt=receipt;this.signed=null;this.changed();
   try {
    const response=await this.services.send(signed);
    if(response.txid!==receipt.txid)throw Error('Node response transaction ID mismatch.');
    this.message='Submitted, not confirmed. Explicitly check confirmation.';
    return receipt;
   } catch(e) {this.message=`OUTCOME UNKNOWN — ${receipt.txid}. Never resend or mint again. ${e.message}`;throw Error(this.message);}
  }finally {this.busy=false;this.changed();}
 }
 async confirm() {
  if(this.busy)throw Error('An operation is pending.');
  if(this.recoveryError||!this.receipt)throw Error(this.recoveryError||'No recovery transaction to check.');
  const receipt=this.receipt,expected=restoreReceipt(JSON.stringify(receipt));
  this.confirmedHere=false;
  this.busy=true;this.changed();
  try {
   assertMainnet(await this.services.params());
   const pending=await this.services.pending(expected.txid);
   if(pending.poolError!=='')throw Error(`Unconfirmed/pool error: ${pending.poolError}`);
   if(!(pending.confirmedRound>0n)||!(pending.assetIndex>0n))throw Error('Unconfirmed: no verified creation round and asset ID.');
   if(!pending.txn?.txn||pending.txn.txn.txID()!==expected.txid||!equal(sdk.encodeUnsignedTransaction(pending.txn.txn),from64(expected.unsigned)))throw Error('Confirmed transaction mismatch.');
   const txn=sdk.decodeUnsignedTransaction(from64(expected.unsigned));
   if(pending.confirmedRound<txn.firstValid||pending.confirmedRound>txn.lastValid)throw Error('Confirmation round mismatch.');
   const asset=await this.services.asset(pending.assetIndex),p=asset.params,r=expected.review;
   const text=(plain,encoded)=>encoded ? new TextDecoder('utf-8',{fatal:true}).decode(encoded):plain;
   const role=value=>!value||value===sdk.ALGORAND_ZERO_ADDRESS_STRING?null:value;
   if(asset.index!==pending.assetIndex||p.creator!==r.sender||p.total!==1n||p.decimals!==0||(p.defaultFrozen??false)!==false||text(p.name,p.nameB64)!==r.name||text(p.unitName,p.unitNameB64)!==r.unit||text(p.url,p.urlB64)!==r.url||!absentHash(p.metadataHash)||['manager','reserve','freeze','clawback'].some(k=>role(p[k])!==r[k]))throw Error('Confirmed asset parameters mismatch.');
   const result={assetId:pending.assetIndex.toString(),round:pending.confirmedRound.toString(),txid:expected.txid};
   const confirmed={...expected,status:'confirmed',result};const raw=JSON.stringify(confirmed);
   this.storage.setItem(RECOVERY_KEY,raw);if(this.storage.getItem(RECOVERY_KEY)!==raw)throw Error('Confirmation recovery persistence failed.');
   this.receipt=confirmed;this.confirmedHere=true;this.message=`Confirmed minted — asset ${result.assetId}, round ${result.round}, txid ${result.txid}`;
   return result;
  } catch(e){this.message=`Unconfirmed / outcome unknown — ${expected.txid}: ${e.message}. Recovery lock remains; do not mint again.`;throw e;}
  finally{this.busy=false;this.changed();}
 }
 startAnother() {
  if(this.busy||!this.confirmedHere||this.receipt?.status!=='confirmed')throw Error('Check and verify the existing mint before starting another NFT.');
  this.storage.removeItem(RECOVERY_KEY);if(this.storage.getItem(RECOVERY_KEY)!==null)throw Error('Could not clear confirmed recovery receipt.');
  this.receipt=null;this.confirmedHere=false;this.message='';this.invalidate();
 }
 invalidate() {this.epoch++;this.review=null;this.signed=null;this.changed();}
 capture() {
  const c=this.getContext();
  if(!c.instance||!sdk.isValidAddress(c.sender))throw Error('Connect and select a valid public address.');
  if(!c.bytes||!c.url)throw Error('Generate valid ARC69 JSON and supply a direct media URL first.');
  validateNote(c.bytes);validateMediaURL(c.url);
  return {...c,bytes:c.bytes.slice(),form:{...c.form}};
 }
 guard(snapshot,epoch) {if(epoch!==this.epoch||!sameContext(snapshot,this.getContext()))throw Error('Mint context changed; prepare and review again.');}
 requireReview() {
  if(!this.review)throw Error('Prepare and review first.');
  this.guard(this.review.context,this.review.epoch);
  if(this.now()-this.review.at>120000 || this.now()<this.review.at)throw Error('Review stale; prepare again.');
  return this.review;
 }
 async sign() {
  this.assertNoRecovery();
  if(this.busy)throw Error('An operation is pending.');
  const review=this.requireReview();this.signed=null;this.busy=true;
  try {
   // Deliberately no await/import/network call before the wallet invocation.
   const request=review.context.instance.signTransaction([[{txn:sdk.decodeUnsignedTransaction(review.unsigned),signers:[review.context.sender]}]]);
   this.changed();const result=await request;this.requireReview();
   if(!Array.isArray(result)||result.length!==1||!(result[0] instanceof Uint8Array))throw Error('Pera must return exactly one signed byte payload.');
   const signed=result[0].slice();await validateSigned(signed,review.unsigned);
   this.requireReview();this.signed=signed;return signed;
  } finally {this.busy=false;this.changed();}
 }
 async prepare() {
  this.assertNoRecovery();
  if(this.busy)throw Error('An operation is pending.');
  this.invalidate();const snapshot=this.capture(),epoch=this.epoch;this.busy=true;this.changed();
  try {
   const params=await this.services.params();assertMainnet(params);this.guard(snapshot,epoch);
   const account=await this.services.account(snapshot.sender);this.guard(snapshot,epoch);
   const result=await buildMint({...snapshot,params,account});this.guard(snapshot,epoch);
   this.review={...result,context:snapshot,epoch,at:this.now()};return this.review;
  }finally{this.busy=false;this.changed();}
 }
}
