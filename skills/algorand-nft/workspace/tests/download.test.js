import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {Window} from 'happy-dom';
import {saveBytes} from '../src/download.js';
test('download creates exact-byte JSON Blob and revokes object URL',async()=>{
  const win=new Window(); let blob; let revoked; let clicked=false;
  const api={createObjectURL:value=>{blob=value;return 'blob:local';},revokeObjectURL:value=>{revoked=value;}};
  win.HTMLAnchorElement.prototype.click=function(){clicked=true;assert.equal(this.download,'metadata.json');};
  const bytes=new TextEncoder().encode(' {"x":1}\r\n');
  saveBytes(win.document,'metadata.json',bytes,'application/json',api,fn=>fn());
  assert.equal(clicked,true);assert.equal(revoked,'blob:local');assert.equal(blob.type,'application/json');
  assert.deepEqual(new Uint8Array(await blob.arrayBuffer()),bytes);
  assert.equal(win.document.querySelectorAll('a').length,0);
  await win.happyDOM.close();
});
test('production bootstrap uses static SDK import but delegates construction to explicit actions',async()=>{
  const main=await readFile(new URL('../src/main.js',import.meta.url),'utf8');
  assert.match(main,/import \{PeraWalletConnect\} from '@perawallet\/connect'/);
  assert.match(main,/import \{isValidAddress\} from 'algosdk'/);
  assert.match(main,/mount\(document/);
  assert.doesNotMatch(main,/new PeraWalletConnect|reconnectSession|\.connect\(/);
  const html=await readFile(new URL('../index.html',import.meta.url),'utf8');
  assert.match(html,/MainNet • ARC69 preparation and explicit NFT minting/);
  assert.match(main,/mintServices:createServices\(\)/);
  assert.match(main,/storage:/);
  for(const path of ['app.js','core.js','wallet.js','download.js']){
    const source=await readFile(new URL('../src/'+path,import.meta.url),'utf8');
    assert.doesNotMatch(source,/innerHTML|fetch\(|XMLHttpRequest|signTransaction\(|reconnectSession\(|localStorage|sessionStorage/);
  }
});
