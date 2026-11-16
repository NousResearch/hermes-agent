import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {Window} from 'happy-dom';
import {mount} from '../src/app.js';
async function setup(){
 const window=new Window({settings:{disableComputedStyleRendering:true,disableJavaScriptEvaluation:true,disableJavaScriptFileLoading:true,disableCSSFileLoading:true,disableIframePageLoading:true}});
 window.document.write(await readFile(new URL('../index.html',import.meta.url),'utf8'));
 const calls=[];
 class Fake{constructor(opts){calls.push(opts);}on(){return ()=>{};}connect(){calls.push('connect');return Promise.resolve(['public-address']);}disconnect(){return Promise.resolve();}dispose(){}}
 const app=mount(window.document,{Connector:Fake,validAddress:a=>a==='public-address'});
 const $=id=>window.document.getElementById(id),input=(id,value)=>{$(id).value=value;$(id).dispatchEvent(new window.Event('input',{bubbles:true}));};
 return {window,app,$,input,settle:()=>new Promise(r=>setTimeout(r,20))};
}
test('ARC69 form rejects oversize or invalid JSON properties and renders public text without HTML execution',async()=>{
 const f=await setup(),{$,input}=f;
 for(const [id,value,pattern] of [['description','é'.repeat(600),/1024/],['properties','[]',/object/],['mime_type','not-a-mime',/MIME/]]){
  input(id,value);$('apply-form').click();assert.equal(f.app.doc.bytes,null);assert.match($('error').textContent,pattern);
  input('description','Hermes');input('properties','{}');input('mime_type','image/png');
 }
 input('description','<img src=x onerror=alert(1)>');input('properties','{"artist":"<script>bad</script>"}');$('apply-form').click();
 assert.equal(JSON.parse($('raw').value).description,'<img src=x onerror=alert(1)>');assert.equal(f.window.document.querySelector('img'),null);
 assert.equal($('error').textContent,'');assert.equal($('prepare-mint').disabled,true);
 await f.window.happyDOM.close();
});
test('ARC69 form creates exact compact public note without checkbox, hashes, uploads or automatic wallet calls',async()=>{
 const f=await setup(),{$,input}=f;
 assert.equal(f.window.document.querySelectorAll('input[type=checkbox]').length,0);
 assert.equal($('ardrive').getAttribute('href'),'https://app.ardrive.io');assert.equal($('ardrive').rel,'noopener noreferrer');
 assert.equal($('asset-name').value,'Hermes');assert.equal($('asset-unit').value,'HRMS01');assert.equal($('admin-policy').value,'immutable');
 assert.equal($('asset-url').value,'https://arweave.net/wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4#i');assert.equal($('mime_type').value,'image/png');
 $('apply-form').click();
 const text=new TextDecoder().decode(f.app.doc.bytes),json=JSON.parse(text);
 assert.equal(text,JSON.stringify(json));assert.equal($('raw').value,text);assert.equal(json.standard,'arc69');assert.equal(json.mime_type,'image/png');
 assert.match($('note-size').textContent,new RegExp(`${f.app.doc.bytes.length} / 1024 UTF-8 bytes`));
 assert.equal('name' in json,false);assert.equal('hash' in f.app.doc,false);assert.equal('upload' in f.app.doc,false);
 for(const id of ['same-file','metadata-id','download-metadata','image-file','sign-consent'])assert.equal($(id),null);
 input('description','Edited');assert.equal(f.app.doc.bytes,null);assert.equal($('raw').value,'');
 $('apply-form').click();assert.equal(JSON.parse(new TextDecoder().decode(f.app.doc.bytes)).description,'Edited');
 $('connect').click();await f.settle();assert.equal($('account').value,'public-address');$('disconnect').click();await f.settle();assert.equal($('account').options.length,0);
 await f.window.happyDOM.close();
});
