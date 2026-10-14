#!/usr/bin/env node
// nano-pay.cjs — minimal, auditable Nano (XNO) wallet CLI for the proof-agent skill.
//
// Commands: new | address | balance | receive | fund [amountXno] | send <toAddress> <amountRaw>
//
// Configuration (all optional except NANO_SEED for wallet commands):
//   NANO_SEED      64-hex wallet seed. Required by every command except `new`.
//                  Never log or commit it; store with 600 permissions.
//   NANO_RPC_URLS  Comma-separated Nano RPC endpoints, tried in order.
//                  Defaults to public nodes (no API key needed). Point this at
//                  your own node for privacy/reliability, e.g.
//                  NANO_RPC_URLS=http://127.0.0.1:7076
//   NANO_RPC_KEY   Optional Authorization header value for keyed RPC providers.
//
// Requires Node >= 18 (global fetch). Dependency: nanocurrency-web (pinned in
// package.json next to this file; run `npm ci` here once).

const { wallet, block, tools } = require('nanocurrency-web');

const DEFAULT_RPC_URLS = [
  'https://rainstorm.city/api',
  'https://nanoslo.0x.no/proxy',
  'https://rpc.nano.to',
];
const RPC_URLS = (process.env.NANO_RPC_URLS || DEFAULT_RPC_URLS.join(','))
  .split(',').map((s) => s.trim()).filter(Boolean);
const RPC_KEY = process.env.NANO_RPC_KEY || '';

// Fallback representative, only used for a wallet's first (open) block.
const REP = 'nano_3arg3asgtigae3xckabaaewkx3bzsh7nwz7jkmjos79ihyaxwphhm6qgjps4';
const ZERO_HASH = '0'.repeat(64);
const RAW_PER_XNO = BigInt('1000000000000000000000000000000');
// Network work thresholds: send/change vs receive/open.
const DIFF_SEND = 'fffffff800000000';
const DIFF_RECEIVE = 'fffffe0000000000';

function xnoToRaw(xno) {
  const [whole, frac = ''] = String(xno).split('.');
  return (BigInt(whole || '0') * RAW_PER_XNO
    + BigInt((frac + '0'.repeat(30)).slice(0, 30))).toString();
}

function rawToXno(raw) {
  return Number((BigInt(raw) * 1000000n) / RAW_PER_XNO) / 1e6;
}

async function rpc(action, params = {}, timeoutMs = 12000) {
  const headers = { 'Content-Type': 'application/json' };
  if (RPC_KEY) headers['Authorization'] = RPC_KEY;
  let lastError;
  for (const url of RPC_URLS) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify({ action, ...params }),
        signal: ctrl.signal,
      });
      const data = await res.json();
      if (data.error && data.error !== 'Account not found') throw new Error(data.error);
      return data;
    } catch (err) {
      lastError = err; // try next endpoint
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastError || new Error('all RPC endpoints failed');
}

async function generateWork(hash, difficulty) {
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await rpc('work_generate', { hash, difficulty }, 28000);
      if (res.work) return res.work;
    } catch (err) { /* retry: rpc() already rotated endpoints */ }
  }
  throw new Error('work_generate failed');
}

async function accountInfo(address) {
  const res = await rpc('account_info', {
    account: address, representative: true, pending: true,
  });
  if (res.error === 'Account not found') return null;
  return {
    balance: res.balance || '0',
    frontier: res.frontier || '',
    representative: res.representative || '',
  };
}

function isValidAddress(address) {
  if (!address || (!address.startsWith('nano_') && !address.startsWith('xrb_'))) return false;
  try { return tools.addressToPublicKey(address) !== null; } catch { return false; }
}

function account(seed) {
  if (!/^[0-9a-fA-F]{64}$/.test(seed)) {
    throw new Error('NANO_SEED must be a 64-hex seed (run `new` to create one)');
  }
  return wallet.fromLegacySeed(seed).accounts[0];
}

// Pocket every receivable block. Handles the first-ever receive (open block,
// zero frontier) as well as subsequent receives, updating the frontier in-loop.
async function receiveAll(seed) {
  const acct = account(seed);
  let info = await accountInfo(acct.address);
  const receivedHashes = [];
  for (let pass = 0; pass < 10; pass++) {
    const pending = await rpc('receivable', {
      account: acct.address, count: '10', source: true,
    });
    const entries = Object.entries(pending.blocks || {});
    if (entries.length === 0) break;
    const [blockHash, blockInfo] = entries[0];
    const amount = typeof blockInfo === 'string' ? blockInfo : blockInfo.amount;
    const opened = Boolean(info && info.frontier);
    const workHash = opened ? info.frontier : tools.addressToPublicKey(acct.address);
    const work = await generateWork(workHash, DIFF_RECEIVE);
    const signed = block.receive({
      walletBalanceRaw: opened ? info.balance : '0',
      toAddress: acct.address,
      representativeAddress: (opened && info.representative) || REP,
      frontier: opened ? info.frontier : ZERO_HASH,
      transactionHash: blockHash,
      amountRaw: amount,
      work,
    }, acct.privateKey);
    const processed = await rpc('process', {
      json_block: 'true', subtype: opened ? 'receive' : 'open', block: signed,
    });
    if (processed.hash) {
      receivedHashes.push(processed.hash);
      info = await accountInfo(acct.address);
    }
  }
  return {
    received: receivedHashes.length,
    hashes: receivedHashes,
    balanceRaw: info ? info.balance : '0',
  };
}

async function sendRaw(seed, toAddress, amountRaw) {
  if (!isValidAddress(toAddress)) throw new Error('invalid recipient');
  if (BigInt(amountRaw) <= 0n) throw new Error('amount must be > 0');
  const acct = account(seed);
  let info = await accountInfo(acct.address);
  // Auto-pocket pending funds first if the balance can't cover the send.
  if (!info || BigInt(info.balance) < BigInt(amountRaw)) {
    await receiveAll(seed);
    info = await accountInfo(acct.address);
  }
  if (!info) throw new Error('account unopened / no funds');
  if (BigInt(amountRaw) > BigInt(info.balance)) throw new Error('insufficient balance');
  const work = await generateWork(info.frontier, DIFF_SEND);
  const signed = block.send({
    walletBalanceRaw: info.balance,
    fromAddress: acct.address,
    toAddress,
    representativeAddress: info.representative || REP,
    frontier: info.frontier,
    amountRaw: String(amountRaw),
    work,
  }, acct.privateKey);
  const processed = await rpc('process', { json_block: 'true', subtype: 'send', block: signed });
  if (!processed.hash) throw new Error('process failed');
  return processed.hash;
}

async function main() {
  const [cmd, arg1, arg2] = process.argv.slice(2);
  const seed = process.env.NANO_SEED || '';

  if (cmd === 'new') {
    const fresh = wallet.generateLegacy();
    console.log(JSON.stringify({ seed: fresh.seed, address: fresh.accounts[0].address }));
    return;
  }
  if (cmd === 'address') {
    console.log(account(seed).address);
    return;
  }
  if (cmd === 'balance') {
    const addr = account(seed).address;
    const info = await accountInfo(addr);
    console.log(JSON.stringify({
      address: addr,
      balanceRaw: info ? info.balance : '0',
      balanceXno: info ? rawToXno(info.balance) : 0,
    }));
    return;
  }
  if (cmd === 'receive') {
    console.log(JSON.stringify({ ok: true, ...(await receiveAll(seed)) }));
    return;
  }
  if (cmd === 'fund') {
    const addr = account(seed).address;
    const info = await accountInfo(addr);
    const amountRaw = arg1 ? xnoToRaw(arg1) : '';
    console.log(JSON.stringify({
      address: addr,
      needXno: arg1 || null,
      uri: 'nano:' + addr + (amountRaw ? '?amount=' + amountRaw : ''),
      balanceXno: info ? rawToXno(info.balance) : 0,
      message: 'Ask your owner to fund this address'
        + (arg1 ? ' with about ' + arg1 + ' XNO' : '') + '.',
    }));
    return;
  }
  if (cmd === 'send') {
    const hash = await sendRaw(seed, arg1, arg2);
    console.log(JSON.stringify({ ok: true, hash, to: arg1, amountRaw: arg2 }));
    return;
  }
  console.error('usage: nano-pay.cjs new | address | balance | receive | fund [amountXno] | send <toAddress> <amountRaw>');
  process.exit(1);
}

main().catch((err) => { console.error('ERROR:', err.message); process.exit(1); });
