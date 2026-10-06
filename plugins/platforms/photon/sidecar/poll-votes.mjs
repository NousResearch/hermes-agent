function nonempty(value) {
  return typeof value === "string" && value.trim() ? value : null;
}

export function parsePollVoteId(messageId, senderId) {
  if (typeof messageId !== "string") return null;
  // Parse the fixed suffix from the right; sender handles can contain colons.
  const match = /^(.*):([^:]+):(selected|deselected):(-?\d+)(?::seq:(\d+))?$/.exec(messageId);
  if (!match) return null;
  const prefix = match[1];
  const senderSuffix = nonempty(senderId) ? `:${senderId}` : null;
  const separator = prefix.indexOf(":");
  const pollId = senderSuffix && prefix.endsWith(senderSuffix)
    ? prefix.slice(0, -senderSuffix.length)
    : separator > 0 ? prefix.slice(0, separator) : null;
  return pollId ? { pollId, eventTime: Number(match[4]),
    ...(match[5] ? { sequence: Number(match[5]) } : {}) } : null;
}

// Event ids remembered per poll so a reconnect replay is a no-op.
const SEEN_EVENTS_PER_POLL = 512;

export class PollVoteTracker {
  constructor(limit = 200) {
    this.limit = limit;
    this.polls = new Map();
    // Polls this process saw from creation; their totals are complete.
    this.complete = new Set();
    this.seenPolls = new Set();
  }

  noteCreated(pollId) {
    if (!nonempty(pollId) || this.seenPolls.has(pollId)) return;
    this.complete.delete(pollId);
    this.complete.add(pollId);
    while (this.complete.size > this.limit) this.complete.delete(this.complete.keys().next().value);
  }

  normalize(content, message = {}) {
    const parsed = parsePollVoteId(message.id, message.sender?.id);
    const pollId = [content.pollId, content.pollMessageGuid, content.poll?.id,
      content.poll?.pollMessageGuid, content.poll?.messageId,
      message.pollId, message.pollMessageGuid].map(nonempty).find(Boolean)
      ?? parsed?.pollId ?? null;
    const vote = {
      type: "poll_option",
      title: content.option?.title ?? content.title ?? "",
      selected: content.selected !== false,
      pollTitle: content.poll?.title ?? content.pollTitle ?? null,
      pollId,
      tally: {},
      voters: 0,
      partial: true,
    };
    if (!pollId) return vote;
    let poll = this.polls.get(pollId);
    if (!poll && this.seenPolls.has(pollId)) this.complete.delete(pollId);
    this.seenPolls.add(pollId);
    while (this.seenPolls.size > this.limit * 10) this.seenPolls.delete(this.seenPolls.values().next().value);
    if (!poll) poll = { voters: new Map(), updates: new Map(), options: new Set(), seen: new Set() };
    this.polls.delete(pollId);
    this.polls.set(pollId, poll);
    while (this.polls.size > this.limit) {
      const evicted = this.polls.keys().next().value;
      this.polls.delete(evicted);
      this.complete.delete(evicted);
    }
    vote.partial = !this.complete.has(pollId);
    for (const option of content.poll?.options ?? []) {
      if (nonempty(option?.title)) poll.options.add(option.title);
    }
    if (nonempty(vote.title)) poll.options.add(vote.title);
    const voter = nonempty(message.sender?.id);
    const eventId = nonempty(message.id);
    // A replayed event id was already applied. Re-applying it after a later
    // deselection in the same millisecond would restore the removed vote.
    const replay = eventId !== null && poll.seen.has(eventId);
    if (eventId && !replay) {
      poll.seen.add(eventId);
      if (poll.seen.size > SEEN_EVENTS_PER_POLL) poll.seen.delete(poll.seen.values().next().value);
    }
    if (voter && nonempty(vote.title) && !replay) {
      const selections = poll.voters.get(voter) ?? new Set();
      const updates = poll.updates.get(voter) ?? new Map();
      const order = parsed?.sequence ?? parsed?.eventTime ?? new Date(message.timestamp).getTime();
      const previous = updates.get(vote.title);
      // Sequence orders distinct events even when their timestamps are identical.
      // Legacy ids use time; their unseen equal-time events retain stream order.
      if (!previous || !Number.isFinite(previous.order) || !Number.isFinite(order) || order >= previous.order) {
        if (vote.selected) selections.add(vote.title);
        else selections.delete(vote.title);
        updates.set(vote.title, { order });
        poll.updates.set(voter, updates);
        if (selections.size) poll.voters.set(voter, selections);
        else poll.voters.delete(voter);
      }
    }
    vote.tally = Object.fromEntries([...poll.options].map((title) => [title, 0]));
    for (const selections of poll.voters.values()) {
      for (const title of selections) vote.tally[title] += 1;
    }
    vote.voters = poll.voters.size;
    return vote;
  }
}
