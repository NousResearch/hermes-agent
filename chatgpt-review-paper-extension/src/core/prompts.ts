import type { ReviewTier } from './types';

export function tierInstruction(tier: ReviewTier): string {
  const common = 'Enumerate concrete, specific topics actually discussed. Do not write a vague mood summary. Never invent content not present in the source. All tiers cover the same source data; only density changes.';
  if (tier === 'ume') return `${common}\n梅 / Light: write an 80-120 word review. Include one short digest paragraph and 3-5 one-line headline bullets. No per-conversation breakdown.`;
  if (tier === 'take') return `${common}\n竹 / Medium: write 250-350 words. Group by theme rather than raw conversation. For each theme, give 1-2 sentences on what was discussed, key decisions/conclusions, and open or unresolved threads. End with a 3-item "worth revisiting" list.`;
  return `${common}\n松 / Deep: write 900-1200 words. For each significant conversation include: (1) what was discussed, (2) conclusions/decisions, (3) open questions/unresolved points, (4) review points. Close with 5-10 question→answer active-recall flashcards drawn from the period.`;
}

export function systemPrompt(tier: ReviewTier, language?: string): string {
  return `You generate a single review paper from the user's own ChatGPT history. Output Markdown only. ${language ? `Write in ${language}.` : 'Match the dominant language of the source conversations.'}\n\n${tierInstruction(tier)}`;
}
