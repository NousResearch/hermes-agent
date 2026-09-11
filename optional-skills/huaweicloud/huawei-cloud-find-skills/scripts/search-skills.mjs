import { Buffer } from 'node:buffer';

const INDEX_URL =
  'https://gitcode.com/api/v5/repos/2501_91318609/skills-for-index/contents/skills-index/index.json?ref=main';
const CN_EN_MAP_URL =
  'https://gitcode.com/api/v5/repos/2501_91318609/skills-for-index/contents/skills-index/cn-en-map.json?ref=main';

const GENERIC_KEYWORDS = new Set([
  '华为云',
  'huawei',
  'huawei cloud',
  '云',
  'cloud',
  '技能',
  'skill',
  'skills',
  '所有',
  'all',
  '全部',
  '有什么',
  '有哪些',
  '相关',
  '列表',
  'list',
  '查找',
  '搜索',
  '发现',
  '浏览',
  'find',
  'search',
  'discover',
  'browse',
  'show',
  'explore',
  'agent',
  '市场',
  'market',
  '类目',
  'category',
  '安装',
  'install',
]);

async function fetchJson(url, _label) {
  const resp = await fetch(url, { headers: { 'User-Agent': 'huaweicloud-devkit/1.0' } });
  const data = await resp.json();
  if (data?.encoding === 'base64' && data.content) {
    return JSON.parse(Buffer.from(data.content, 'base64').toString('utf8'));
  }
  return data;
}

function isGeneric(kw) {
  const k = kw.toLowerCase().trim();
  return GENERIC_KEYWORDS.has(k) || /华为云|huawei/i.test(k);
}

function expandKeywords(raw, cnEnMap) {
  if (!raw) return [[], []];
  const parts = raw.replace(/[,;]/g, ' ').split(/\s+/).filter(Boolean);
  const expanded = [...parts];
  for (const p of parts) {
    const lower = p.toLowerCase();
    if (cnEnMap[p]) expanded.push(cnEnMap[p]);
    for (const [cn, en] of Object.entries(cnEnMap)) {
      if (en === lower) expanded.push(cn);
    }
  }
  const unique = [...new Set(expanded)].sort();
  return [unique.filter((kw) => !isGeneric(kw)), unique.filter((kw) => isGeneric(kw))];
}

function scoreSkill(skill, specificKws, genericKws) {
  if (!specificKws.length && !genericKws.length) return [1, []];
  let total = 0;
  const matched = [];
  const nl = (skill.name || '').toLowerCase();
  const dl = (skill.description || '').toLowerCase();
  const sl = (skill.service || '').toLowerCase();
  const tl = (skill.triggers || []).map((t) => (t || '').toLowerCase());

  for (const kw of specificKws) {
    const k = kw.toLowerCase();
    let s = 0;
    if (nl.includes(k)) s += 10;
    else if (tl.some((t) => t.includes(k))) s += 8;
    else if (dl.includes(k)) s += 5;
    else if (sl.includes(k)) s += 3;
    if (s > 0) {
      total += s;
      matched.push(kw);
    }
  }
  for (const kw of genericKws) {
    const k = kw.toLowerCase();
    let s = 0;
    if (nl.includes(k)) s += 10;
    else if (tl.some((t) => t.includes(k))) s += 4;
    else if (dl.includes(k)) s += 2;
    else if (sl.includes(k)) s += 1;
    if (s > 0) {
      total += s;
      matched.push(kw);
    }
  }
  if (!specificKws.length && total === 0) {
    total = 1;
    if (dl.length > 20) total += 1;
    if (tl.length) total += 1;
  }
  return [total, matched];
}

async function main() {
  const keyword =
    process.argv
      .slice(2)
      .filter((a) => !a.startsWith('-'))
      .join(' ') || '';
  const catIdx = process.argv.indexOf('-c');
  const category = catIdx >= 0 ? process.argv[catIdx + 1] || '' : '';

  try {
    const [idx, cnEnMap] = await Promise.all([fetchJson(INDEX_URL, 'index'), fetchJson(CN_EN_MAP_URL, 'cn-en-map')]);

    if (!keyword && !category) {
      console.log(`Categories: ${(idx.categories || []).join(', ')}`);
      console.log('Usage: node search-skills.mjs <keyword> [-c <category>]');
      return;
    }

    const [specificKws, genericKws] = expandKeywords(keyword, cnEnMap);
    const hasSpecific = specificKws.length > 0;

    const results = [];
    for (const skill of idx.skills || []) {
      if (category && skill.category !== category) continue;
      const [score, matched] = scoreSkill(skill, specificKws, genericKws);
      if (hasSpecific && score === 0) continue;
      const desc = (skill.description || '').slice(0, 150);
      results.push({
        score,
        name: skill.name,
        category: skill.category,
        service: skill.service,
        description: desc + (skill.description?.length > 150 ? '...' : ''),
        triggers: (skill.triggers || []).slice(0, 5),
        matched,
      });
    }
    results.sort((a, b) => b.score - a.score);

    if (!results.length) {
      console.log(`No results. Try broader keywords, remove category filter, or switch CN↔EN.`);
      return;
    }

    console.log(`Found ${results.length} skill(s):`);
    for (const r of results) {
      console.log(
        `  [${r.score}pts] ${r.name} (${r.category}/${r.service})${r.matched.length ? ' matched: ' + r.matched.join(',') : ''}`,
      );
      console.log(`    ${r.description}`);
    }
  } catch (e) {
    console.error(`Error: ${e.message}`);
    process.exit(1);
  }
}

main();
