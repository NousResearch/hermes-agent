// Send-time poll option identifiers, kept across sidecar restarts.
//
// Photon can omit option identifiers from both the live `created` delta and
// `client.polls.get()`. Only the `polls.create()` response carries them, so a
// restart between sending a poll and the first vote would otherwise leave the
// vote unmappable. The patched SDK (patch-spectrum-mixed-attachments.mjs)
// calls remember() on send and get() on a cache miss through
// `globalThis.__hermesPhotonPollSeeds`.
import fs from "node:fs";
import path from "node:path";

function nonempty(value) {
  return typeof value === "string" && value.trim() ? value : null;
}

export class PollSeedStore {
  constructor(file = null, limit = 200) {
    this.file = nonempty(file);
    this.limit = limit;
    this.polls = new Map();
  }

  load() {
    if (!this.file) return this;
    let saved;
    try {
      saved = JSON.parse(fs.readFileSync(this.file, "utf8"));
    } catch (err) {
      if (err?.code !== "ENOENT") {
        console.error(`photon-sidecar: WARNING ignoring unreadable poll seed file: ${err?.message || err}`);
      }
      return this;
    }
    for (const poll of Array.isArray(saved?.polls) ? saved.polls : []) {
      const options = (Array.isArray(poll?.options) ? poll.options : [])
        .filter((option) => nonempty(option?.text) && nonempty(option?.optionIdentifier))
        .map(({ text, optionIdentifier }) => ({ text, optionIdentifier }));
      if (nonempty(poll?.id) && options.length) {
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
    return {
      poll: { title: seed.title || "Poll", options },
      optionsByIdentifier: new Map(seed.options.map(({ optionIdentifier }, index) => [optionIdentifier, options[index]])),
    };
  }

  has(id) {
    return this.polls.has(id);
  }

  remember(id, title, options) {
    const known = (Array.isArray(options) ? options : [])
      .filter((option) => nonempty(option?.text) && nonempty(option?.optionIdentifier))
      .map(({ text, optionIdentifier }) => ({ text, optionIdentifier }));
    if (!nonempty(id) || !known.length) return;
    this.polls.delete(id);
    this.polls.set(id, { title: nonempty(title) ?? "", options: known });
    this.#trim();
    this.#save();
  }

  #trim() {
    while (this.polls.size > this.limit) this.polls.delete(this.polls.keys().next().value);
  }

  // Temp file + rename: a crash mid-write keeps the previous complete file.
  #save() {
    if (!this.file) return;
    const temp = `${this.file}.${process.pid}.tmp`;
    try {
      fs.mkdirSync(path.dirname(this.file), { recursive: true });
      const polls = [...this.polls].map(([id, seed]) => ({ id, ...seed }));
      fs.writeFileSync(temp, JSON.stringify({ version: 1, polls }), { encoding: "utf8", mode: 0o600 });
      fs.renameSync(temp, this.file);
    } catch (err) {
      try {
        fs.rmSync(temp, { force: true });
      } catch {
        // The directory itself is unusable; nothing was written.
      }
      console.error(`photon-sidecar: WARNING could not save poll seeds: ${err?.message || err}`);
    }
  }
}
