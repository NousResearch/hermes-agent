/**
 * Intent checks: did the rewrite ask the agent for something the user did not?
 * Keeping every exact word (preservation.ts) does not keep the intent, so this
 * compares what the draft and the rewrite ask the agent to DO, by category:
 *
 *   destructive   delete, overwrite, drop a table, rm -rf, force-push…
 *   file-change   edits to files or code when the draft only asked to read,
 *                 explain or review
 *   install       installs, new dependencies, upgrades
 *   external      email, messages, publishing, uploads, pushes, deploys
 *   spending      purchases, payments, subscriptions, money amounts
 *   permission    sudo, chmod, credentials, keys, disabling security
 *   delegation    subagents, swarms, delegation: added, dropped or flipped
 *   from-context  a run of words copied from the chat excerpt, not the draft
 *
 * A category is flagged when the rewrite asks for it (an occurrence not
 * negated in its own clause: "do not delete" is not a delete) and the draft
 * does not. Delegation is flagged on any difference, either way.
 *
 * Every finding is review-only: Automatic stops and shows the review, and
 * Approve needs the explicit "I checked" acknowledgement. These are keyword
 * heuristics over English text. They catch the common ways a rewrite widens a
 * request; they cannot prove the meaning is unchanged, and a rewrite can
 * still change intent in words no list anticipates. See README › Limitations.
 */

export type IntentKind =
  'delegation' | 'destructive' | 'external' | 'file-change' | 'from-context' | 'install' | 'permission' | 'spending'

export const INTENT_KINDS: ReadonlySet<string> = new Set<IntentKind>([
  'delegation',
  'destructive',
  'external',
  'file-change',
  'from-context',
  'install',
  'permission',
  'spending'
])

export interface IntentIssue {
  kind: IntentKind
  /** The words that triggered it, as the rewrite (or the draft) has them. */
  value: string
}

interface Rule {
  kind: Exclude<IntentKind, 'from-context'>
  /** What, in the rewrite, asks for this kind of action. */
  trigger: RegExp
  /** What, in the draft, already asked for it (defaults to `trigger`). */
  allowedBy?: RegExp
  /** Flag a difference either way (dropped as well as added). */
  symmetric?: boolean
}

const words = (alternatives: string[]) => alternatives.join('|')

const RULES: readonly Rule[] = [
  {
    kind: 'destructive',
    trigger: new RegExp(
      `\\b(?:${words([
        'delet(?:e|es|ed|ing|ion)',
        'eras(?:e|es|ed|ing)',
        'wip(?:e|es|ed|ing)',
        'purg(?:e|es|ed|ing)',
        'destroy\\w*',
        'truncat\\w*',
        'overwrit\\w*',
        'remov(?:e|es|ed|ing)',
        'uninstall\\w*',
        'discard\\w*',
        'drop (?:the |a |any |all )?(?:tables?|databases?|schemas?|columns?|index(?:es)?|collections?)',
        'reset --hard',
        'git clean',
        'force[- ]push\\w*',
        'rm -[a-z]*[rf][a-z]*'
      ])})|--force\\b`,
      'gi'
    )
  },
  {
    kind: 'file-change',
    trigger: new RegExp(
      `\\b(?:${words([
        'modif(?:y|ies|ied|ying)',
        'edit(?:s|ed|ing)? (?:the |any |your |these |those )?(?:files?|code|source|config\\w*)',
        'refactor\\w*',
        'renam(?:e|es|ed|ing)',
        'rewrit(?:e|es|ing) (?:the |any |your )?(?:files?|code|modules?|functions?)',
        'apply (?:the |a |your )?(?:patch|fix|changes?|diff)',
        'git commit',
        'commit (?:the |your |these |all |any )?changes',
        'write (?:it |them |the \\w+ )?(?:to|into) (?:a |the )?(?:files?|disk)',
        'save (?:it |them |the \\w+ )?(?:to|as) (?:a |the )?files?',
        'fix(?:es|ed|ing)? (?:the |any |all |it\\b|them\\b|bugs?|issues?|errors?)',
        'implement(?:s|ed|ing)?\\b',
        'patch (?:the|it)\\b'
      ])})`,
      'gi'
    ),
    // A draft that asks for any change at all already allows changes; this
    // only catches a read-only request (explain, review, summarize) turned
    // into one that edits.
    allowedBy: new RegExp(
      `\\b(?:${words([
        'writ(?:e|es|ing)',
        'creat\\w*',
        'add(?:s|ed|ing)?',
        'build\\w*',
        'mak(?:e|es|ing)',
        'generat\\w*',
        'fix\\w*',
        'updat\\w*',
        'chang\\w*',
        'edit\\w*',
        'modif\\w*',
        'implement\\w*',
        'refactor\\w*',
        'renam\\w*',
        'mov(?:e|es|ing)',
        'commit\\w*',
        'sav(?:e|es|ing)',
        'apply\\w*',
        'patch\\w*',
        'rewrit\\w*',
        'replac\\w*',
        'convert\\w*',
        'migrat\\w*',
        'set ?up',
        'configur\\w*',
        'install\\w*',
        'delet\\w*',
        'remov\\w*',
        'clean\\w*',
        'format\\w*',
        'optimi[sz]\\w*',
        'improv\\w*',
        'draft\\w*',
        'produc\\w*'
      ])})\\b`,
      'gi'
    )
  },
  {
    kind: 'install',
    trigger: new RegExp(
      `\\b(?:${words([
        '(?:re)?install(?:s|ed|ing|ation)?',
        '(?:pip3?|npm|pnpm|yarn|brew|apt|apt-get|cargo|uv|gem|go) (?:install|add|get|i)\\b',
        'add (?:a |an |the |new |extra )?(?:dependenc(?:y|ies)|packages?|librar(?:y|ies)|plugins?)',
        'upgrad(?:e|es|ed|ing) (?:the |all |any )?(?:packages?|dependenc(?:y|ies)|librar(?:y|ies)|system)'
      ])})`,
      'gi'
    )
  },
  {
    kind: 'external',
    trigger: new RegExp(
      `\\b(?:${words([
        'e-?mail(?:s|ed|ing)?',
        'send (?:(?:it|this|them|the|a|an|out)\\s+)?(?:e-?mail|message|dm|text|notification|invite|reply|newsletter)s?',
        'post(?:s|ed|ing)? (?:it |this |them )?(?:to|on) ',
        'publish(?:es|ing)?\\b',
        'tweet\\w*',
        'slack',
        'discord',
        'telegram',
        'whatsapp',
        'webhooks?',
        'upload(?:s|ed|ing)? (?:it|them|this|that|everything|the \\w+|your \\w+|to)\\b',
        'share (?:it|this|them|the \\w+) (?:with|publicly|online|on)',
        'open (?:a |an )?(?:pull request|pr|issue)\\b',
        'file (?:a |an )?(?:issue|ticket|bug report)',
        'git push',
        'push (?:it |them |the \\w+ |your \\w+ )?(?:to|upstream)\\b',
        'deploy(?:s|ed|ing)?\\b',
        'notif(?:y|ies|ied|ying)',
        'contact (?:the|a|an|your|support)\\b',
        'reach out'
      ])})`,
      'gi'
    )
  },
  {
    kind: 'spending',
    trigger: new RegExp(
      `\\b(?:${words([
        'buy(?:s|ing)?',
        'bought',
        'purchas\\w*',
        'pay(?:s|ing)?',
        'paid',
        'payments?',
        'subscrib\\w*',
        'subscriptions?',
        'checkout',
        'billing',
        'spend(?:s|ing)? (?:money|budget|credits?|funds)',
        'order (?:a|an|the|more|new)',
        '(?:upgrade|switch) (?:to )?(?:a |the |your )?(?:paid|pro|premium|enterprise) (?:plan|tier|account|version)'
      ])})\\b|(?<![\\w$])[$€£¥]\\s?(?:\\d{2,}|\\d(?:[.,]\\d+)+)`,
      'gi'
    )
  },
  {
    kind: 'permission',
    trigger: new RegExp(
      `\\b(?:${words([
        'sudo',
        'chmod',
        'chown',
        'root (?:access|privileges|user)',
        'admin(?:istrator)? (?:access|rights|privileges|role)',
        'grant\\w* (?:\\w+ )?(?:access|permissions?|privileges)',
        // Doing something with a secret or a permission, not naming one: "where
        // credentials are stored" describes, "paste your API key" acts.
        '(?:add|use|set|create|generate|rotate|share|paste|print|log|expose|commit|store|save|send|change|reset|give|provide|enter|hardcode|embed) (?:\\w+ ){0,2}(?:credentials?|api[- ]?keys?|access[- ]tokens?|secrets?|passwords?|ssh[- ]keys?|private[- ]keys?|permissions?|privileges?)',
        'disabl\\w* (?:the )?(?:firewall|sandbox|security|authentication|auth|2fa|mfa|ssl|tls|verification|checks?)',
        'bypass\\w*',
        'elevat\\w* (?:privileges|permissions|access)'
      ])})\\b|--no-verify\\b|--dangerously[\\w-]*`,
      'gi'
    )
  },
  {
    kind: 'delegation',
    trigger: new RegExp(
      `\\b(?:${words([
        'swarms?',
        'sub-?agents?',
        'delegat\\w*',
        'spawn\\w* (?:\\w+ )?agents?',
        'parallel agents?',
        'multi-?agents?',
        'multiple agents',
        'other agents?',
        'another agent',
        'hand(?:s|ed|ing)? (?:it |this |them )?off'
      ])})\\b`,
      'gi'
    ),
    symmetric: true
  }
]

/** A word in the same clause before the match that turns it into a
 *  prohibition or a precondition ("do not delete", "ask before deleting"). */
const NEGATED_BEFORE =
  /\b(?:not|no|never|without|avoid(?:ing)?|nor|cannot|instead of|rather than|refrain from|(?:ask|confirm|check)\w*(?: \w+){0,3} (?:before|prior to))\b|n['’]t\b/i

/** Where a clause ends, so a negation in an earlier clause does not cover a
 *  later instruction ("don't touch the tests, but delete the fixtures"). */
const CLAUSE_BREAK = /[.;!?\n]|\bbut\b|\bthen\b|\bhowever\b/gi

/** A quantity right after the verb that cancels it ("install no new
 *  dependencies", "change nothing"). */
const NEGATED_AFTER = /^\s+(?:no|nothing|none|zero)\b/i

const NEGATION_LOOKBACK = 60

function isNegated(text: string, index: number, length: number): boolean {
  const before = text.slice(Math.max(0, index - NEGATION_LOOKBACK), index)
  const clause = before.split(CLAUSE_BREAK).pop() ?? ''

  return NEGATED_BEFORE.test(clause) || NEGATED_AFTER.test(text.slice(index + length))
}

interface Occurrences {
  /** First occurrence that asks for the action. */
  asked: null | string
  /** First occurrence that forbids it. */
  forbidden: null | string
}

function occurrences(text: string, re: RegExp): Occurrences {
  const found: Occurrences = { asked: null, forbidden: null }

  for (const match of text.matchAll(re)) {
    const value = match[0].trim()

    if (isNegated(text, match.index ?? 0, match[0].length)) {
      found.forbidden ??= value
    } else {
      found.asked ??= value
    }

    if (found.asked && found.forbidden) {
      break
    }
  }

  return found
}

function ruleIssue(rule: Rule, original: string, candidate: string): IntentIssue | null {
  const added = occurrences(candidate, rule.trigger).asked

  if (!rule.symmetric) {
    return added && !occurrences(original, rule.allowedBy ?? rule.trigger).asked
      ? { kind: rule.kind, value: added }
      : null
  }

  const before = occurrences(original, rule.trigger)
  const after = occurrences(candidate, rule.trigger)

  if (Boolean(before.asked) !== Boolean(after.asked)) {
    return { kind: rule.kind, value: (after.asked ?? before.asked) as string }
  }

  if (Boolean(before.forbidden) !== Boolean(after.forbidden)) {
    return { kind: rule.kind, value: (after.forbidden ?? before.forbidden) as string }
  }

  return null
}

/** How many words in a row, copied from the chat excerpt and absent from the
 *  draft, count as the rewrite carrying the conversation into the prompt. */
export const CONTEXT_COPY_WORDS = 8

const tokens = (text: string) => text.toLowerCase().match(/[\p{L}\p{N}_'’-]+/gu) ?? []

function shingles(list: readonly string[]): Set<string> {
  const out = new Set<string>()

  for (let i = 0; i + CONTEXT_COPY_WORDS <= list.length; i += 1) {
    out.add(list.slice(i, i + CONTEXT_COPY_WORDS).join(' '))
  }

  return out
}

function contextCopy(original: string, candidate: string, context: string): IntentIssue | null {
  if (!context.trim()) {
    return null
  }

  const fromContext = shingles(tokens(context))
  const fromDraft = shingles(tokens(original))

  for (const span of shingles(tokens(candidate))) {
    if (fromContext.has(span) && !fromDraft.has(span)) {
      return { kind: 'from-context', value: span }
    }
  }

  return null
}

/** What the candidate asks for that the draft did not (see the module doc). */
export function checkIntent(original: string, candidate: string, context = ''): IntentIssue[] {
  const issues = RULES.map(rule => ruleIssue(rule, original, candidate))
  issues.push(contextCopy(original, candidate, context))

  return issues.filter((issue): issue is IntentIssue => issue !== null)
}
