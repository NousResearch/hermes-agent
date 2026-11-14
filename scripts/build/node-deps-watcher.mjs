// The lock watcher (node-deps-lock.mjs runWatched): kills the child the relay started, with
// everything it started, once the lock is no longer `token`'s. Started by the relay in a session
// of its own; the child's pid arrives on stdin once the relay has started it, and the watcher ends
// with the relay (stdin closes). Arguments: lock, token, poll interval in ms.
import { execFileSync, spawnSync } from 'node:child_process'
import { readdirSync, readFileSync } from 'node:fs'

const [lock, token, every] = process.argv.slice(2);
let root;
let pending = false;
const lost = () => { try { return !readFileSync(lock, 'utf8').includes(token) } catch (error) { return error.code === 'ENOENT' } };
function parents() {
  const table = new Map();
  if (process.platform === 'linux') {
    for (const pid of readdirSync('/proc')) {
      if (!/^[0-9]+$/.test(pid)) continue;
      try {
        const stat = readFileSync('/proc/' + pid + '/stat', 'utf8');
        table.set(Number(pid), Number(stat.slice(stat.lastIndexOf(')') + 2).split(' ')[1]));
      } catch {
        // Exited while the table was read.
      }
    }
  } else {
    for (const line of execFileSync('ps', ['-A', '-o', 'pid=,ppid='], { encoding: 'utf8' }).split('\n')) {
      const [pid, ppid] = line.trim().split(/\s+/).map(Number);
      if (pid) table.set(pid, ppid);
    }
  }
  return table;
}
function tree() {
  let table;
  try { table = parents() } catch { return new Set([Number(root)]) }
  const found = new Set([Number(root)]);
  for (let grew = true; grew;) {
    grew = false;
    for (const [pid, ppid] of table) if (found.has(ppid) && !found.has(pid)) { found.add(pid); grew = true }
  }
  return found;
}
const signal = (pid, name) => {
  try {
    process.kill(pid, name)
  } catch {
    // Already gone.
  }
};
function end() {
  if (process.platform === 'win32') {
    spawnSync('taskkill', ['/pid', root, '/T', '/F'], { stdio: 'ignore' });
  } else {
    const seen = new Set();
    for (let fresh = [Number(root)]; fresh.length; fresh = [...tree()].filter(pid => !seen.has(pid))) {
      for (const pid of fresh) { seen.add(pid); signal(pid, 'SIGSTOP') }
    }
    for (const pid of seen) signal(pid, 'SIGKILL');
  }
  process.exit(0);
}
const tick = () => { if (lost()) { if (root) end(); else pending = true } };
tick();
setInterval(tick, Number(every));
let input = '';
process.stdin.on('data', data => {
  input += data;
  if (root === undefined && input.includes('\n')) { root = input.split('\n')[0]; if (pending) end() }
}).on('end', () => process.exit(0));
