import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkGfm from 'remark-gfm';
import remarkDirective from 'remark-directive';
import { parse } from 'yaml';
import { remarkPreserveDirectives } from '../remark/preserve-directives.mjs';

/** Honor authored descriptions; otherwise use the first prose paragraph. */

const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/;
// Characters that join without spaces in CJK text (soft line breaks must not
// inject one); Latin boundaries keep their spacing.
const CJK = '[\\u3000-\\u9fff\\uf900-\\ufaff\\uff00-\\uffef]';

/** MDX ESM statements and brace expressions render nothing, but a CommonMark
 * parse sees them as text paragraphs — recognize them so they never win. */
function isStatement(paragraph) {
  const first = paragraph.children?.[0];
  return first?.type === 'text' && /^\s*(?:import|export)\s/.test(first.value);
}

function isExpression(paragraph) {
  return paragraph.children?.length === 1
    && paragraph.children[0].type === 'text'
    && /^\{[\s\S]*\}$/.test(paragraph.children[0].value.trim());
}

/** Plain prose of a phrasing subtree: text and inline code render visibly;
 * images, inline HTML/JSX and nested expressions contribute nothing. */
function proseText(paragraph) {
  const parts = [];
  const visit = (node) => {
    if (node.type === 'break') parts.push(' ');
    else if (node.type === 'text' || node.type === 'inlineCode') parts.push(node.value);
    else if (Array.isArray(node.children)) node.children.forEach(visit);
  };
  visit(paragraph);
  return parts
    .join('')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(new RegExp(`(${CJK}) (?=${CJK})`, 'g'), '$1');
}

/** First prose paragraph of a markdown body: headings, fences, raw HTML/JSX
 * blocks, tables, lists and admonition containers are not lead prose. */
function firstProseParagraph(body) {
  const tree = unified().use(remarkParse).use(remarkGfm).use(remarkDirective).parse(body);
  for (const node of tree.children) {
    if (node.type !== 'paragraph' || isStatement(node) || isExpression(node)) continue;
    remarkPreserveDirectives()(node);
    const prose = proseText(node);
    if (prose) return prose;
  }
  return undefined;
}

export function documentDescription(text) {
  const match = FRONTMATTER.exec(text);
  if (match) {
    const data = parse(match[1]) ?? {};
    if (typeof data.description === 'string' && data.description.trim()) return data.description.trim();
  }
  return firstProseParagraph(match ? text.slice(match[0].length) : text);
}
