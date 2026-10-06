function nonempty(value) {
  return typeof value === "string" && value.trim() ? value : null;
}

export function parsePollVoteId(messageId, senderId) {
  if (typeof messageId !== "string") return null;
  // Parse the fixed suffix from the right; sender handles can contain colons.
  const match = /^(.*):([^:]+):(selected|deselected):(-?\d+)$/.exec(messageId);
  if (!match) return null;
  const prefix = match[1];
  const senderSuffix = nonempty(senderId) ? `:${senderId}` : null;
  const separator = prefix.indexOf(":");
  const pollId = senderSuffix && prefix.endsWith(senderSuffix)
    ? prefix.slice(0, -senderSuffix.length)
    : separator > 0 ? prefix.slice(0, separator) : null;
  return pollId ? { pollId, eventTime: Number(match[4]) } : null;
}

export class PollVoteTracker {
  constructor(limit = 200) {
    this.limit = limit;
    this.polls = new Map();
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
    };
    if (!pollId) return vote;
    let poll = this.polls.get(pollId);
    if (!poll) poll = { voters: new Map(), updates: new Map(), options: new Set() };
    this.polls.delete(pollId);
    this.polls.set(pollId, poll);
    while (this.polls.size > this.limit) this.polls.delete(this.polls.keys().next().value);
    for (const option of content.poll?.options ?? []) {
      if (nonempty(option?.title)) poll.options.add(option.title);
    }
    if (nonempty(vote.title)) poll.options.add(vote.title);
    const voter = nonempty(message.sender?.id);
    if (voter && nonempty(vote.title)) {
      const selections = poll.voters.get(voter) ?? new Set();
      const updates = poll.updates.get(voter) ?? new Map();
      const time = parsed?.eventTime ?? new Date(message.timestamp).getTime();
      const previous = updates.get(vote.title);
      // Reconnects can replay a selection after its deselection. Keep the latest state.
      if (!previous || !Number.isFinite(previous.time) || !Number.isFinite(time) || time >= previous.time) {
        if (vote.selected) selections.add(vote.title);
        else selections.delete(vote.title);
        updates.set(vote.title, { time });
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
