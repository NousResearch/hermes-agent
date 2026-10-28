import { resolveDocumentLink } from '../lib/content.mjs';

function walk(node, visit) {
  visit(node);
  if (node.children) for (const child of node.children) walk(child, visit);
}

/** GitHub-readable source links become app routes, not deployed file URLs. */
export function remarkDocumentLinks({ contentDir }) {
  return (tree, file) => {
    walk(tree, (node) => {
      if (node.type === 'link' || node.type === 'definition') {
        node.url = resolveDocumentLink(file.path, node.url, contentDir);
      }
    });
  };
}

/** Docusaurus also accepted unlabeled fences and these readable aliases. */
export function remarkCodeLanguages() {
  const aliases = { ascii: 'text', listing: 'text', excel: 'text', conf: 'text', cron: 'text', jsonl: 'json', env: 'dotenv', txt: 'text', plaintext: 'text' };
  return (tree) => walk(tree, (node) => {
    if (node.type === 'code' && node.lang in aliases) node.lang = aliases[node.lang];
  });
}
