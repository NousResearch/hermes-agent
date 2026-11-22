// The relay node-deps-lock.mjs runWatched blocks on: starts the lock watcher, then the child that
// writes the trees once the watcher has started, and exits with the child's status (128 + n for a
// signal). Arguments: lock, token, poll interval in ms, then the child's file and arguments.
import { spawn } from 'node:child_process'
import { constants } from 'node:os'
import { fileURLToPath } from 'node:url'

const [lock, token, every, file, ...args] = process.argv.slice(2);
let signalled = false;
for (const name of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(name, () => { signalled = true });
const parent = process.ppid;
// The watcher starts first: a child that writes the trees never runs unwatched.
const watch = spawn(process.execPath, [fileURLToPath(new URL('./node-deps-watcher.mjs', import.meta.url)), lock, token, every],
  { stdio: ['pipe', 'ignore', 'inherit'], detached: process.platform !== 'win32', windowsHide: true });
watch.on('error', error => {
  console.error('node-deps: cannot start the lock watcher, so ' + file + ' did not run (' + error.message + ')');
  process.exit(127);
});
watch.on('spawn', () => {
  const child = spawn(file, args, { stdio: 'inherit' });
  setInterval(() => { if (!signalled && process.ppid !== parent) { signalled = true; child.kill('SIGTERM') } }, 200);
  child.on('error', error => { console.error(error.message); process.exit(127) });
  // A watcher that has died since (EPIPE) must not crash the relay and leave the child unreaped.
  watch.stdin.on('error', () => {});
  if (child.pid) watch.stdin.write(child.pid + '\n');
  child.on('exit', (code, signal) => process.exit(code ?? 128 + (constants.signals[signal] ?? 0)));
});
