import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../src/core.js';
const encode=s=>new TextEncoder().encode(s);
test('standard alone is sufficient and all optional ARC69 fields can be omitted',()=>{
 assert.equal(new TextDecoder().decode(core.formBytes({})),'{"standard":"arc69"}');
 assert.deepEqual(core.fromForm({properties:'{}'}),{standard:'arc69'});
});
test('note cap is UTF-8 bytes, accepts exactly 1024 and rejects 1025 or multibyte overflow',()=>{
 const overhead=core.formBytes({description:'x'}).length-1;
 assert.equal(core.formBytes({description:'x'.repeat(1024-overhead)}).length,1024);
 assert.throws(()=>core.formBytes({description:'x'.repeat(1025-overhead)}),/1024/);
 const description='é'.repeat(600);assert.ok(description.length<1024);
 assert.throws(()=>core.formBytes({description}),/1024/);
});
test('invalid standard, field types, properties, BOM, UTF-8 and JSON fail closed',()=>{
 for(const value of [null,[],{}, {standard:'arc3'},{standard:'ARC69'},{standard:'arc69',name:'ASA only'},{standard:'arc69',description:1},{standard:'arc69',properties:[]},{standard:'arc69',properties:null},{standard:'arc69',mime_type:'png'}])assert.throws(()=>core.validateNote(encode(JSON.stringify(value))));
 for(const properties of ['null','[]','not json'])assert.throws(()=>core.formBytes({properties}));
 for(const bytes of [new Uint8Array(),new Uint8Array([239,187,191,...encode('{"standard":"arc69"}')]),new Uint8Array([255]),encode('{')])assert.throws(()=>core.validateNote(bytes));
});
test('direct media references preserve literal IDs and URLs, reject ARC3 and unsafe inputs',()=>{
 const id='wmocohO4PYO5wxtwPOOEfPBwwTBZP8GgIzqCz1qWtF4';
 assert.deepEqual(core.mediaRef(id),{id,url:`https://arweave.net/${id}`});
 for(const url of [`https://arweave.net/${id}#i`,'ipfs://bafyexample/image#i','https://example.org/image.png?version=1'])assert.equal(core.mediaRef(url).url,url);
 for(const url of [id+'x',id.slice(1),' '+id,id+' ','http://example.org/a','https://u:p@example.org/a','https://example.org/a#arc3','javascript:alert(1)','https://example.org/a\\b'])assert.throws(()=>core.mediaRef(url));
 const prefix='https://example.org/';assert.equal(core.validateMediaURL(prefix+'x'.repeat(96-prefix.length)).length,96);
 assert.throws(()=>core.validateMediaURL(prefix+'x'.repeat(97-prefix.length)),/96/);
});
test('optional ARC69 URLs support external page anchors without treating them as ASA type tags',()=>{
 const note=core.formBytes({external_url:'https://example.org/art#details',media_url:'ipfs://bafyexample/media#v',mime_type:'video/mp4'});
 assert.equal(core.validateNote(note).external_url,'https://example.org/art#details');
 assert.throws(()=>core.formBytes({media_url:'https://example.org/movie#arc3'}),/fragment/);
});
test('form generates compact ARC69 onchain JSON, optional properties and no ASA name fields',()=>{
 const bytes=core.formBytes({description:'Hermes',mime_type:'image/png',properties:'{"artist":"Hermes"}'});
 assert.equal(new TextDecoder().decode(bytes),'{"standard":"arc69","description":"Hermes","mime_type":"image/png","properties":{"artist":"Hermes"}}');
 assert.deepEqual(core.validateNote(bytes),JSON.parse(new TextDecoder().decode(bytes)));
 assert.equal('name' in core.fromForm({name:'ASA only'}),false);
});
