// Send-time poll option identifiers, kept across sidecar restarts.
// Photon can omit these from both live deltas and polls.get(). The patched SDK
// reads this store through globalThis.__hermesPhotonPollSeeds.
import fs from "node:fs";
import path from "node:path";
import { randomUUID } from "node:crypto";

const MAX_SEED_BYTES = 1024 * 1024;
const SENT_POLL_LIMIT = 2000;
const STALE_LOCK_MS = 10_000;

function nonempty(value) {
  return typeof value === "string" && value.trim() ? value : null;
}

function trim(map, limit) {
  while (map.size > limit) map.delete(map.keys().next().value);
}

export class PollSeedStore {
  constructor(file = null, limit = 200) {
    this.file = nonempty(file);
    this.limit = limit;
    this.polls = new Map();
    this.sentPollIds = new Map();
  }

  load() {
    if (!this.file) return this;
    let saved, fd;
    try {
      fd = fs.openSync(this.file, "r");
      if (fs.fstatSync(fd).size > MAX_SEED_BYTES) throw new Error("oversized poll seed file");
      // Bound the read even if another writer grows the file after stat().
      const bytes = Buffer.alloc(MAX_SEED_BYTES + 1);
      const length = fs.readSync(fd, bytes, 0, bytes.length, 0);
      if (length > MAX_SEED_BYTES) throw new Error("oversized poll seed file");
      saved = JSON.parse(bytes.subarray(0, length).toString("utf8"));
    } catch (err) {
      if (err?.code !== "ENOENT") {
        console.error(`photon-sidecar: WARNING ignoring unreadable poll seed file: ${err?.message || err}`);
      }
      return this;
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
    }
    for (const id of Array.isArray(saved?.sentPollIds) ? saved.sentPollIds : []) {
      if (nonempty(id)) this.sentPollIds.set(id, true);
    }
    for (const poll of Array.isArray(saved?.polls) ? saved.polls : []) {
      const options = (Array.isArray(poll?.options) ? poll.options : [])
        .filter((option) => nonempty(option?.text))
        .map(({ text, optionIdentifier }) => ({ text, optionIdentifier }));
      if (nonempty(poll?.id) && options.some((option) => nonempty(option.optionIdentifier))) {
        this.sentPollIds.set(poll.id, true);
        this.polls.set(poll.id, { title: nonempty(poll.title) ?? "", options });
      }
    }
    this.#trim();
    return this;
  }

  // Shaped like the SDK's cached poll: { poll: {title, options}, optionsByIdentifier }.
  get(id) {
    const seed = this.polls.get(id);
    if (!seed) return undefined;
    const options = seed.options.map(({ text }) => ({ title: text }));
    const identifiers = seed.options.flatMap(({ optionIdentifier }, index) =>
      nonempty(optionIdentifier) ? [[optionIdentifier, options[index]]] : []);
    return {
      poll: { type: "poll", title: seed.title || "Poll", options },
      optionsByIdentifier: new Map(identifiers),
    };
  }

  has(id) {
    return this.polls.has(id);
  }

  remember(id, title, options) {
    if (!nonempty(id)) return;
    const known = (Array.isArray(options) ? options : [])
      .filter((option) => nonempty(option?.text))
      .map(({ text, optionIdentifier }) => ({ text, optionIdentifier }));
    this.sentPollIds.delete(id);
    this.sentPollIds.set(id, true);
    if (known.some((option) => nonempty(option.optionIdentifier))) {
      this.polls.delete(id);
      this.polls.set(id, { title: nonempty(title) ?? "", options: known });
    }
    this.#trim();
    this.#save(id);
  }

  #trim() {
    trim(this.polls, this.limit);
    trim(this.sentPollIds, SENT_POLL_LIMIT);
  }

  #lock() {
    const lock = `${this.file}.lock`;
    for (let attempt = 0; attempt < 100; attempt++) {
      try {
        return fs.openSync(lock, "wx", 0o600);
      } catch (err) {
        if (err?.code !== "EEXIST") throw err;
        // A save holds the lock for milliseconds, so an old lock belongs to a killed writer.
        try {
          if (Date.now() - fs.statSync(lock).mtimeMs > STALE_LOCK_MS) {
            fs.rmSync(lock, { force: true });
            continue;
          }
        } catch (statErr) {
          if (statErr?.code !== "ENOENT") throw statErr;
          continue;
        }
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 10);
      }
    }
    throw new Error("poll seed writer lock busy");
  }

  // Serialize merge + rename across processes so two sidecars cannot lose a send.
  #save(id) {
    if (!this.file) return;
    const temp = `${this.file}.${process.pid}.${randomUUID()}.tmp`;
    let lock;
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
      lock = this.#lock();
      const disk = new PollSeedStore(this.file, this.limit).load();
      const newest = this.polls.get(id);
      this.polls = new Map([...disk.polls, ...[...this.polls].filter(([pollId]) => !disk.polls.has(pollId))]);
      this.sentPollIds = new Map([...disk.sentPollIds, ...this.sentPollIds]);
      if (newest) {
        this.polls.delete(id);
        this.polls.set(id, newest);
      }
      this.sentPollIds.delete(id);
      this.sentPollIds.set(id, true);
      this.#trim();
      const polls = [...this.polls].map(([pollId, seed]) => ({ id: pollId, ...seed }));
      const saved = JSON.stringify({ version: 1, polls, sentPollIds: [...this.sentPollIds.keys()] });
      if (Buffer.byteLength(saved) > MAX_SEED_BYTES) throw new Error("oversized poll seeds to save");
      fs.writeFileSync(temp, saved, { encoding: "utf8", flag: "wx", mode: 0o600 });
      fs.renameSync(temp, this.file);
      fs.chmodSync(this.file, 0o600);
    } catch (err) {
      console.error(`photon-sidecar: WARNING could not save poll seeds: ${err?.message || err}`);
    } finally {
      try {
        fs.rmSync(temp, { force: true });
      } catch (err) {
        console.error(`photon-sidecar: WARNING could not clean poll seed temp file: ${err?.message || err}`);
      }
      if (lock !== undefined) {
        fs.closeSync(lock);
        fs.rmSync(`${this.file}.lock`, { force: true });
      }
    }
  }
}
