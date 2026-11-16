import {formBytes} from './core.js';
import {WalletController} from './wallet.js';
import {mountMint} from './mint-ui.js';

export function mount(document,{Connector,validAddress,mintServices,storage}) {
 const $=id=>document.getElementById(id);
 const doc={bytes:null,version:0};let mintUI;
 const error=value=>{$('error').textContent=value?.message||String(value||'');};
 const wallet=new WalletController(Connector,validAddress,()=>{
  mintUI?.invalidate();
  $('connect').disabled=wallet.busy||!!wallet.instance;$('disconnect').disabled=!wallet.instance;
  $('account').replaceChildren();
  for(const address of wallet.accounts){const option=document.createElement('option');option.value=address;option.textContent=address;$('account').append(option);}
  $('account').value=wallet.selected;$('account').disabled=!wallet.accounts.length;
  $('wallet-status').textContent=wallet.error||(wallet.selected?'MainNet requested, wallet-approved connection. Node identity checked during preparation.':wallet.busy?'Waiting for wallet approval…':'Not connected.');
 });
 function invalidate(){
  doc.version++;doc.bytes=null;$('raw').value='';$('note-size').textContent='No note generated.';mintUI?.invalidate();
 }
 $('connect').onclick=()=>wallet.connect();$('disconnect').onclick=()=>wallet.disconnect();
 $('account').onchange=()=>{try{wallet.select($('account').value);}catch(e){error(e);}};
 $('metadata-form').onsubmit=event=>event.preventDefault();
 const fields=['description','external_url','media_url','mime_type','properties'];
 for(const id of fields){$(id).addEventListener('input',invalidate);$(id).addEventListener('change',invalidate);}
 $('apply-form').onclick=()=>{
  invalidate();
  try {
   const form=Object.fromEntries(fields.map(id=>[id,$(id).value]));doc.bytes=formBytes(form);
   $('raw').value=new TextDecoder().decode(doc.bytes);$('note-size').textContent=`${doc.bytes.length} / 1024 UTF-8 bytes (application cap)`;error('');
  }catch(e){error(e);}
  mintUI?.render();
 };
 mintUI=mountMint(document,{wallet,doc,services:mintServices,storage:storage||{getItem:()=>null,setItem:()=>{throw Error('Recovery storage unavailable.');},removeItem:()=>{throw Error('Recovery storage unavailable.');}}});
 return {doc,wallet,mint:mintUI.controller};
}
