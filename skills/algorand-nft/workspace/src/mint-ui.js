import {MintController,describeMint} from './mint.js';
import {mediaRef} from './core.js';
export function mountMint(document,{wallet,doc,services,storage}) {
 const $=id=>document.getElementById(id);
 const controller=new MintController({getContext:()=>({instance:wallet.instance,generation:wallet.generation,sender:wallet.selected,version:doc.version,bytes:doc.bytes,url:mediaRef($('asset-url').value).url,form:{name:$('asset-name').value,unit:$('asset-unit').value,policy:$('admin-policy').value}}),services,storage,changed:render});
 function render() {
  const locked=!!controller.receipt||!!controller.recoveryError,busy=controller.busy;
  $('prepare-mint').disabled=!services||busy||locked||!doc.bytes||!wallet.selected;
  $('sign-mint').disabled=busy||locked||!controller.review;
  $('broadcast-mint').disabled=busy||locked||!controller.signed;
  $('check-confirmation').disabled=!services||busy||!controller.receipt||!!controller.recoveryError;
  $('start-another').disabled=busy||!!controller.recoveryError||!controller.confirmedHere;
  $('mint-review').textContent=controller.review?JSON.stringify(describeMint(controller.review.unsigned),null,2):'No transaction prepared.';
  $('mint-recovery').textContent=controller.receipt?JSON.stringify(controller.receipt,null,2):controller.recoveryError||'No recovery receipt.';
  $('mint-status').textContent=controller.recoveryError||controller.message||(controller.receipt?'Saved submission outcome unverified. Confirm this exact transaction ID; never resend.':controller.signed?'Verified signed bytes held in memory only. Not broadcast.':busy?'Operation pending…':'Not submitted.');
 }
 async function act(operation){try{await operation();$('error').textContent='';}catch(e){$('error').textContent=e?.message||String(e);}finally{render();}}
 const invalidate=()=>controller.invalidate();
 for(const id of ['asset-name','asset-unit','asset-url','admin-policy']){$(id).addEventListener('input',invalidate);$(id).addEventListener('change',invalidate);}
 $('prepare-mint').onclick=()=>act(()=>controller.prepare());
 $('sign-mint').onclick=()=>act(()=>controller.sign());
 $('broadcast-mint').onclick=()=>act(()=>controller.broadcast());
 $('check-confirmation').onclick=()=>act(()=>controller.confirm());
 $('start-another').onclick=()=>act(()=>controller.startAnother());
 render();return {controller,invalidate,render};
}
