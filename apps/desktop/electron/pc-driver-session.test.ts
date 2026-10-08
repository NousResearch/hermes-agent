import assert from 'node:assert/strict'
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import { PcDriverSession } from './pc-driver-session'

test('MCP initialization, tool result and shutdown preserve request identity', async () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-pc-test-'))
  const fixture = path.join(dir, 'fake.cjs')
  writeFileSync(fixture, `const readline=require('node:readline');const rl=readline.createInterface({input:process.stdin});rl.on('line',line=>{const m=JSON.parse(line);if(m.id)process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:m.id,result:m.method==='initialize'?{protocolVersion:'2024-11-05',capabilities:{},serverInfo:{name:'fake',version:'1'}}:{content:[{type:'text',text:m.params.name}]}})+'\\n')});rl.on('close',()=>process.exit(0));`)
  const driver = new PcDriverSession(process.execPath, [fixture])
  try {
    assert.deepEqual(await driver.call('list_windows', {}), { content: [{ type: 'text', text: 'list_windows' }] })
    driver.close()
    await assert.rejects(driver.call('click', {}), /closed/)
  } finally { driver.close(); rmSync(dir, { recursive: true, force: true }) }
})

test('driver loss fails pending action without replay', async () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-pc-test-'))
  const fixture = path.join(dir, 'fake.cjs')
  writeFileSync(fixture, `const readline=require('node:readline');readline.createInterface({input:process.stdin}).on('line',line=>{const m=JSON.parse(line);if(m.method==='initialize')process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:m.id,result:{}})+'\\n');if(m.method==='tools/call')process.exit(1)});`)
  const driver = new PcDriverSession(process.execPath, [fixture])
  try { await assert.rejects(driver.call('click', {}), /unknown.*Do not replay/) }
  finally { driver.close(); rmSync(dir, { recursive: true, force: true }) }
})
