#!/usr/bin/env node
/**
 * CodeGraph Evidence Adapter — LAH Repo Router
 * 
 * Normalizes CodeGraph structural evidence for repository disambiguation.
 * Called only when the LAH Repo Router sets codegraph_required=true.
 * 
 * Input (stdin or argv): JSON with:
 *   { mission, candidate_repos, requested_roles: { implementation, memory, context } }
 * 
 * Output (stdout): Normalized structured evidence JSON.
 * 
 * Phases:
 *   1. Derive bounded query set from mission nouns
 *   2. Check index freshness for each candidate
 *   3. Run codegraph explore/query on each candidate
 *   4. Collect evidence in 6 categories
 *   5. Score each candidate
 *   6. Produce recommendation
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ========== Config ==========
const CODEGRAPH_BINARY = '/home/deploy/.npm-global/bin/codegraph';
const CODEGRAPH_TIMEOUT_MS = 15000;      // 15s per query
const MAX_QUERIES_PER_REPO = 4;
const MAX_FILES_PER_EXPLORE = 5;

// ========== Scoring model (Phase 7) ==========
const SCORES = {
  EXACT_SYMBOL_OR_ROUTE:     30,
  DIRECT_TEST_COVERAGE:      20,
  PRIMARY_FILE_PATH_OWNERSHIP: 20,
  DIRECT_DEPENDENCY_PROVIDER: 15,
  PACKAGE_SERVICE_OWNERSHIP:  15,
  IMPORT_OR_CONSUMER_ONLY:     5,
  DOCUMENTATION_MENTION_ONLY:  2,
  DOES_NOT_OWN_RULE:         -25,
  CONTEXT_ONLY_REPOSITORY:   -15,
  NO_STRUCTURAL_RESULT:      -10,
};

// ========== Confidence thresholds (Phase 8) ==========
const THRESHOLDS = {
  HIGH_SCORE:     60,
  HIGH_LEAD:      20,
  MEDIUM_SCORE:   45,
  MEDIUM_LEAD:    15,
};

// ========== Repository path map (from canonical repos) ==========
const REPO_PATHS = {
  'lah-stack-tools':         '/home/deploy/lah-stack-repos/lah-stack-tools',
  'cartelogic-v2':           '/home/deploy/lah-stack-repos/cartelogic-v2',
  'openclaw-runtime':        '/home/deploy/lah-stack-repos/openclaw-runtime',
  'lah-brain':               '/home/deploy/lah-stack-repos/lah-brain',
  'lah-core':                '/home/deploy/lah-stack-repos/lah-core',
  'lah-discovery-platform':  '/home/deploy/lah-stack-repos/lah-discovery-platform',
  'lah-stack-biz-assets':    '/home/deploy/lah-stack-repos/lah-stack-biz-assets',
  'hermes-agent':            '/home/deploy/hermes-agent',
  'cartelogic-remote-memory':'/home/deploy/cartelogic-remote-memory',
};

// ========== CodeGraph tool wrappers ==========

function runCodegraph(args, path) {
  try {
    const start = Date.now();
    const output = execSync(
      CODEGRAPH_BINARY + ' ' + args.join(' ') + ' -p ' + JSON.stringify(path),
      { encoding: 'utf8', timeout: CODEGRAPH_TIMEOUT_MS }
    );
    return { output, duration_ms: Date.now() - start, exit_code: 0 };
  } catch (e) {
    return {
      output: e.stdout || '',
      stderr: e.stderr || e.message,
      duration_ms: 0,
      exit_code: e.status || 1,
      error: e.message
    };
  }
}

function checkIndexFreshness(repoId) {
  const rp = REPO_PATHS[repoId];
  if (!rp || !fs.existsSync(rp)) return { status: 'NOT_FOUND', detail: 'Repository path does not exist' };

  try {
    const raw = execSync(CODEGRAPH_BINARY + ' status ' + JSON.stringify(rp) + ' --json', {
      encoding: 'utf8', timeout: 5000
    });
    const st = JSON.parse(raw);
    if (!st.initialized) return { status: 'MISSING', detail: 'Not initialized' };
    if (!st.lastIndexed) return { status: 'MISSING', detail: 'No lastIndexed timestamp' };
    if (st.fileCount === 0) return { status: 'NOT_APPLICABLE', detail: 'Index exists but empty (0 files) — likely content-only repo' };

    const indexedDate = new Date(st.lastIndexed);
    const now = Date.now();
    const ageDays = (now - indexedDate.getTime()) / 86400000;

    // Check if repo has changes since index
    const hasChanges = st.pendingChanges &&
      (st.pendingChanges.added > 0 || st.pendingChanges.modified > 0 || st.pendingChanges.removed > 0);

    let status = 'READY';
    const notes = [];
    if (hasChanges) {
      status = 'STALE';
      notes.push('pending changes since index');
    }
    if (ageDays > 14) {
      if (status === 'READY') status = 'STALE';
      notes.push('index is ' + Math.round(ageDays) + ' days old');
    }
    if (st.index && st.index.reindexRecommended) {
      if (status === 'READY') status = 'STALE';
      notes.push('reindex recommended by CodeGraph');
    }

    return {
      status,
      index: st,
      lastIndexed: st.lastIndexed,
      fileCount: st.fileCount,
      nodeCount: st.nodeCount,
      pendingChanges: st.pendingChanges,
      notes: notes.length > 0 ? notes : [],
    };
  } catch (e) {
    return { status: 'BROKEN', detail: e.message, error: e };
  }
}

// ========== Query derivation (Phase 5) ==========

function deriveQueries(mission, repoId, requestedRoles) {
  const ml = mission.toLowerCase();
  const words = ml.split(/[\s_:—–-]+/).filter(w => w.length > 2);
  const uniqueWords = [...new Set(words)];

  // Remove common stopwords
  const stopwords = ['the', 'and', 'for', 'with', 'from', 'this', 'that', 'dans',
    'une', 'des', 'les', 'sur', 'dans', 'pour', 'avec', 'pas', 'qui', 'que',
    'dans', 'est', 'sont', 'fait', 'faire'];
  const significant = uniqueWords.filter(w => !stopwords.includes(w) && w.length > 2);

  // Named concepts from the mission
  const namedConcepts = [];

  // Domain-specific concept extraction
  const conceptCues = {
    'cloe': 'CLOE',
    'gateway': 'gateway',
    'wordpress': 'wordpress',
    'plugin': 'plugin',
    'route': 'route',
    'memory': 'memory',
    'observation': 'observation',
    'operator': 'operator',
    'skill': 'skill',
    'launch': 'start',
    'start': 'start',
    'guard': 'guard',
    'mission': 'mission',
    'tool': 'tool',
    'handler': 'handler',
    'schema': 'schema',
    'test': 'test',
    'deploy': 'deploy',
    'recovery': 'recovery',
    'dashboard': 'dashboard',
    'adapter': 'adapter',
    'provider': 'provider',
    'service': 'service',
    'config': 'config',
    'receipt': 'receipt',
    'feature': 'feature',
    'snippet': 'snippet',
  };

  for (const w of significant) {
    if (conceptCues[w]) {
      namedConcepts.push(conceptCues[w]);
    } else if (w.match(/^[A-Z]/) || w === w.toUpperCase()) {
      // Acronyms and proper nouns
      namedConcepts.push(w.toUpperCase());
    }
  }

  // Also extract any embedded repo/routing concepts
  const missionNouns = significant.filter(w => {
    // Keep longer words as potential component/function names
    return w.length >= 4 && !['this', 'that', 'with', 'from', 'dans'].includes(w);
  });

  // Build query set — bounded to MAX_QUERIES_PER_REPO
  const queries = [];

  // Query 1: The whole mission as a single query (trimmed for clarity)
  const shortMission = mission.replace(/[—–]/g, ' ').trim().substring(0, 80);
  queries.push(shortMission);

  // Query 2: Named concepts joined
  if (namedConcepts.length > 0) {
    queries.push(namedConcepts.slice(0, 4).join(' '));
  }

  // Query 3: Mission nouns (most significant)
  if (missionNouns.length > 0) {
    queries.push(missionNouns.slice(0, 4).join(' '));
  }

  // Query 4: Repo-specific context
  const repoContext = getRepoContextQuery(repoId, mission);
  if (repoContext) {
    queries.push(repoContext);
  }

  // Remove duplicates and empty
  return [...new Set(queries.filter(q => q && q.length > 3))].slice(0, MAX_QUERIES_PER_REPO);
}

function getRepoContextQuery(repoId, mission) {
  // Add repo-specific context based on what the repo owns
  const contexts = {
    'lah-stack-tools': ['mission', 'launch', 'operator', 'tool', 'guard'],
    'cartelogic-v2': ['memory', 'observation', 'context', 'decision'],
    'openclaw-runtime': ['gateway', 'route', 'cloe', 'mvp', 'runtime'],
    'lah-brain': ['business', 'knowledge', 'route', 'web'],
    'lah-core': ['wordpress', 'plugin', 'php', 'snippet'],
    'lah-discovery-platform': ['adr', 'specification', 'design', 'infra'],
    'lah-stack-biz-assets': ['content', 'draft', 'niche', 'profile'],
    'hermes-agent': ['agent', 'skill', 'framework', 'dashboard'],
    'cartelogic-remote-memory': ['memory', 'server', 'snapshot'],
  };
  const ctx = contexts[repoId];
  if (!ctx) return null;
  const missionWords = mission.toLowerCase().split(/[\s_]+/);
  const intersection = ctx.filter(c => missionWords.some(w => w.includes(c)));
  if (intersection.length > 0) {
    return intersection.join(' ');
  }
  return null;
}

// ========== Evidence collection (Phase 6) ==========

async function collectEvidence(repoId, mission, queries) {
  const rp = REPO_PATHS[repoId];
  if (!rp || !fs.existsSync(rp)) {
    return { repository_id: repoId, error: 'Path not found', codegraph_status: 'NOT_FOUND' };
  }

  // Check freshness first
  const freshness = checkIndexFreshness(repoId);

  // Initialize evidence containers
  const matchedFiles = [];
  const matchedSymbols = [];
  const matchedTests = [];
  const dependencyEvidence = [];
  const ownershipEvidence = [];
  const negativeEvidence = [];
  const queryReceipts = [];

  if (freshness.status === 'NOT_APPLICABLE' || freshness.status === 'MISSING') {
    return {
      repository_id: repoId,
      codegraph_status: freshness.status,
      score: 0,
      matched_files: [],
      matched_symbols: [],
      matched_tests: [],
      dependency_evidence: [],
      ownership_evidence: [],
      negative_evidence: [],
      query_receipts: [],
      freshness,
    };
  }

  const isStale = freshness.status === 'STALE';

  // Run queries
  for (const query of queries) {
    const receipt = { query, result: null, error: null };

    try {
      // Try explore first
      const exploreResult = runCodegraph(
        ['explore', JSON.stringify(query), '--max-files', String(MAX_FILES_PER_EXPLORE)],
        rp
      );

      if (exploreResult.exit_code === 0 && exploreResult.output) {
        receipt.result = 'explore_ok';
        receipt.duration_ms = exploreResult.duration_ms;

        // Parse explore output for evidence
        const output = exploreResult.output;

        // Extract file references
        const fileMatches = output.match(/([a-zA-Z0-9_\-./]+\.(?:mjs|js|py|ts|php|yaml|json)):\d+/g);
        if (fileMatches) {
          for (const fm of fileMatches) {
            const fp = fm.replace(/:\d+$/, '');
            if (!matchedFiles.find(f => f.path === fp)) {
              const cls = classifyFile(fp, repoId, mission);
              matchedFiles.push({ path: fp, classification: cls });
            }
          }
        }

        // Check for test files
        const testMatches = fileMatches ? fileMatches.filter(f =>
          f.includes('/test/') || f.includes('.test.') || f.includes('/tests/')
        ) : [];
        for (const tm of (testMatches || [])) {
          const tp = tm.replace(/:\d+$/, '');
          if (!matchedTests.find(t => t.path === tp)) {
            matchedTests.push({ path: tp });
          }
        }

        // Check for symbol definitions
        const symbolPattern = /`([a-zA-Z_$][a-zA-Z0-9_$]*)`\s*\(/g;
        let symbolMatch;
        while ((symbolMatch = symbolPattern.exec(output)) !== null) {
          const sym = symbolMatch[1].replace(/\u001b\[\d+m/g, '');
          if (!matchedSymbols.find(s => s.name === sym) && sym.length > 2) {
            matchedSymbols.push({ name: sym, context: query.substring(0, 60) });
          }
        }

        queryReceipts.push({
          query,
          type: 'explore',
          status: 'ok',
          duration_ms: exploreResult.duration_ms,
          found_files: fileMatches ? fileMatches.length : 0,
        });

        // Also try `codegraph query` for symbol lookup
        const queryResult = runCodegraph(['query', JSON.stringify(query), '--limit', '10'], rp);
        if (queryResult.exit_code === 0 && queryResult.output) {
          const clean = queryResult.output.replace(/\u001b\[\d+m/g, '');
          const qLines = clean.split('\n').filter(l =>
            l.includes('function') || l.includes('class') || l.includes('route')
          );
          for (const ql of qLines) {
            const parts = ql.trim().split(/\s+/);
            const kind = parts[0];
            const name = parts.length > 1 ? parts.slice(1).join(' ').split(/\s+/)[0] : null;
            if (name && name !== query && name.length > 2 && !matchedSymbols.find(s => s.name === name)) {
              matchedSymbols.push({ name, context: kind, source: 'query' });
            }
          }
          queryReceipts.push({
            query,
            type: 'query',
            status: 'ok',
            duration_ms: queryResult.duration_ms,
          });
        }

        // Check for dependency/provider patterns
        if (output.includes('depend') || output.includes('import') || output.includes('require')) {
          dependencyEvidence.push({ type: 'dependency_found', evidence_sample: output.substring(0, 200) });
        }

        // Check for test/verify patterns
        if (output.includes('test') || output.includes('spec') || (exploreResult.output || '').includes('Test')) {
          // Already captured in testMatches above
        }

        // Check for ownership patterns (package.json, service files, etc.)
        if (matchedFiles.some(f => f.path.includes('package.json') || f.path.includes('service'))) {
          ownershipEvidence.push({ type: 'ownership_manifest', detail: 'Found service/package definition files' });
        }
      } else {
        receipt.error = exploreResult.error || 'No output from explore';
        receipt.status = 'error';
      }
    } catch (e) {
      receipt.error = e.message;
      receipt.status = 'error';
    }

    queryReceipts.push(receipt);
  }

  // ========== Negative evidence check ==========

  // Check if this repo is documented as "does_not_own" for the mission concepts
  const missionWords = mission.toLowerCase().split(/[\s_:—–-]+/);
  // We get does_not_own info from the mapping later, but add a structural check
  if (matchedFiles.length === 0 && matchedSymbols.length === 0) {
    negativeEvidence.push({ type: 'no_structural_match', detail: 'No files or symbols matched in CodeGraph queries' });
  }

  // Check: is this repo a content-only repo (markdown, json)?
  const contentExtensions = ['.md', '.txt', '.json', '.meta.json'];
  const matchedContentFiles = matchedFiles.filter(f =>
    contentExtensions.some(ext => f.path.endsWith(ext))
  );
  const matchedCodeFiles = matchedFiles.filter(f =>
    !contentExtensions.some(ext => f.path.endsWith(ext))
  );

  if (matchedContentFiles.length > 0 && matchedCodeFiles.length === 0 && matchedSymbols.length === 0) {
    negativeEvidence.push({
      type: 'documentation_only',
      detail: 'Only documentation/content files matched, no code implementation found'
    });
  }

  // ========== Scoring (Phase 7) ==========

  let score = 0;

  // Symbol matches — highest value
  score += matchedSymbols.length * SCORES.EXACT_SYMBOL_OR_ROUTE;

  // Test coverage
  score += matchedTests.length * SCORES.DIRECT_TEST_COVERAGE;

  // File matches — classify by path
  const codeFileCount = matchedCodeFiles.length;
  const contentFileCount = matchedContentFiles.length;

  // Code files in primary directories
  const primaryPaths = matchedFiles.filter(f =>
    !f.path.includes('/test/') && !f.path.includes('/tests/') && !contentExtensions.some(e => f.path.endsWith(e))
  );
  score += primaryPaths.length * 5;

  // Content-only matches (low value)
  score += contentFileCount * SCORES.DOCUMENTATION_MENTION_ONLY;

  // Dependency/provider evidence
  score += dependencyEvidence.length * SCORES.DIRECT_DEPENDENCY_PROVIDER;

  // Ownership evidence
  score += ownershipEvidence.length * SCORES.PACKAGE_SERVICE_OWNERSHIP;

  // Negative evidence penalties
  for (const ne of negativeEvidence) {
    if (ne.type === 'documentation_only') {
      score += SCORES.DOCUMENTATION_MENTION_ONLY;  // already counted as low value
    }
    if (ne.type === 'no_structural_match') {
      score += SCORES.NO_STRUCTURAL_RESULT;
    }
  }

  // Stale index penalty: cap at medium confidence
  const maxConfidence = isStale ? 'medium' : 'high';

  return {
    repository_id: repoId,
    codegraph_status: freshness.status,
    score,
    matched_files: matchedFiles,
    matched_symbols: matchedSymbols.slice(0, 20), // cap for output
    matched_tests: matchedTests.slice(0, 10),
    dependency_evidence: dependencyEvidence,
    ownership_evidence: ownershipEvidence,
    negative_evidence: negativeEvidence,
    query_receipts: queryReceipts.slice(0, 8),
    freshness,
    max_confidence: maxConfidence,
  };
}

function classifyFile(filePath, repoId, mission) {
  const lower = filePath.toLowerCase();
  const missionLower = mission.toLowerCase();

  if (lower.includes('/test/') || lower.includes('.test.') || lower.includes('/tests/')) return 'test';
  if (lower.endsWith('.md') || lower.endsWith('.txt')) return 'documentation';
  if (lower.includes('package.json') || lower.includes('service')) return 'ownership';
  if (lower.includes('import') || lower.includes('require')) return 'dependency';

  // Mission-relevance check
  const missionWords = missionLower.split(/[\s_:—–-]+/).filter(w => w.length > 3);
  const matches = missionWords.filter(w => lower.includes(w));
  if (matches.length > 0) return 'implementation_relevant';

  return 'implementation';
}

// ========== Recommendation (Phase 8) ==========

function recommend(evidenceList, requestedRoles) {
  if (evidenceList.length === 0) {
    return { recommended_repo: null, confidence: 'none', explanation: 'No evidence collected' };
  }

  // Sort by score descending
  const sorted = [...evidenceList].sort((a, b) => b.score - a.score);

  const top = sorted[0];
  const runnerUp = sorted[1];

  let confidence = 'none';
  let explanation = [];

  if (top.score >= THRESHOLDS.HIGH_SCORE) {
    if (runnerUp && (top.score - runnerUp.score) >= THRESHOLDS.HIGH_LEAD) {
      // Check for strong implementation signal
      const hasImplSignal = top.matched_symbols.length > 0 ||
        top.matched_files.some(f => f.classification === 'implementation_relevant' || f.classification === 'implementation');
      if (hasImplSignal && top.max_confidence === 'high') {
        confidence = 'high';
        explanation.push('Strong structural evidence: top score ' + top.score +
          ' with lead ' + (top.score - (runnerUp ? runnerUp.score : 0)) +
          ' and implementation signal');
      } else if (hasImplSignal) {
        confidence = 'medium';
        explanation.push('Good evidence but index is stale (max_confidence: ' + top.max_confidence + ')');
      }
    } else if (runnerUp) {
      confidence = 'medium';
      explanation.push('Top score ' + top.score + ' but lead ' +
        (top.score - runnerUp.score) + ' below threshold ' + THRESHOLDS.HIGH_LEAD);
    } else {
      confidence = 'high';
      explanation.push('Single candidate with score ' + top.score);
    }
  } else if (top.score >= THRESHOLDS.MEDIUM_SCORE) {
    if (runnerUp && (top.score - runnerUp.score) >= THRESHOLDS.MEDIUM_LEAD) {
      confidence = 'medium';
      explanation.push('Medium evidence: score ' + top.score + ' with adequate lead');
    } else if (runnerUp) {
      confidence = 'low';
      explanation.push('Scores too close: ' + top.score + ' vs ' + runnerUp.score);
    } else {
      confidence = 'low';
      explanation.push('Score ' + top.score + ' below high threshold');
    }
  } else {
    if (top.score < 0 && runnerUp && runnerUp.score > 0) {
      // Top candidate has negative evidence, runner-up is better
      const reordered = sorted.filter(r => r.score > 0);
      if (reordered.length > 0) {
        return recommend([...evidenceList.filter(e => e.repository_id !== top.repository_id)], requestedRoles);
      }
    }
    explanation.push('Insufficient structural evidence: highest score is ' + top.score);
  }

  return {
    recommended_repo: top.score > 0 ? top.repository_id : null,
    confidence,
    explanation: explanation.join('; '),
    scores: evidenceList.map(e => ({
      repository_id: e.repository_id,
      score: e.score,
      codegraph_status: e.codegraph_status,
    })),
  };
}

// ========== Main ==========

async function main() {
  // Read input
  let input = '';
  if (process.argv[2] && process.argv[2] !== '--stdin') {
    // Read JSON from file
    input = fs.readFileSync(process.argv[2], 'utf8');
  } else {
    // Read from stdin
    const chunks = [];
    const reader = process.stdin;
    reader.setEncoding('utf8');
    for await (const chunk of reader) {
      chunks.push(chunk);
    }
    input = chunks.join('');
  }

  let params;
  try {
    params = JSON.parse(input);
  } catch (e) {
    console.error(JSON.stringify({
      status: 'ERROR',
      error: 'Invalid JSON input: ' + e.message,
    }));
    process.exit(1);
  }

  const { mission, candidate_repos, requested_roles } = params;

  if (!mission || !candidate_repos || !Array.isArray(candidate_repos)) {
    console.error(JSON.stringify({
      status: 'ERROR',
      error: 'Missing required fields: mission, candidate_repos (array)',
    }));
    process.exit(1);
  }

  const roles = requested_roles || { implementation: true, memory: false, context: false };

  const startTime = Date.now();

  // Step 1: Check freshness for all candidates
  const freshnesses = {};
  for (const repoId of candidate_repos) {
    freshnesses[repoId] = checkIndexFreshness(repoId);
  }

  // Step 2: Determine which repos can be queried
  const queryable = candidate_repos.filter(repoId => {
    const f = freshnesses[repoId];
    return f && (f.status === 'READY' || f.status === 'STALE');
  });

  // Step 3: Collect evidence per repo
  const evidenceList = [];
  for (const repoId of queryable) {
    const queries = deriveQueries(mission, repoId, roles);
    const evidence = await collectEvidence(repoId, mission, queries);
    evidenceList.push(evidence);
  }

  // Add unavailable repos
  for (const repoId of candidate_repos) {
    if (!queryable.includes(repoId)) {
      evidenceList.push({
        repository_id: repoId,
        codegraph_status: freshnesses[repoId] ? freshnesses[repoId].status : 'NOT_FOUND',
        score: 0,
        matched_files: [],
        matched_symbols: [],
        matched_tests: [],
        dependency_evidence: [],
        ownership_evidence: [],
        negative_evidence: [{ type: 'codegraph_unavailable', detail: freshnesses[repoId] ? freshnesses[repoId].detail : 'Unknown' }],
        query_receipts: [],
        freshness: freshnesses[repoId] || { status: 'NOT_FOUND' },
      });
    }
  }

  // Step 4: Recommend
  const recommendation = recommend(evidenceList, roles);

  const totalMs = Date.now() - startTime;

  // Step 5: Produce output
  const output = {
    status: evidenceList.some(e => e.codegraph_status === 'READY' || e.codegraph_status === 'STALE')
      ? (recommendation.confidence === 'high' || recommendation.confidence === 'medium' ? 'SUCCESS' : 'PARTIAL')
      : 'UNAVAILABLE',
    mission: mission,
    queried_repos: candidate_repos,
    repository_evidence: evidenceList.map(e => ({
      repository_id: e.repository_id,
      codegraph_status: e.codegraph_status,
      score: e.score,
      matched_files: e.matched_files ? e.matched_files.length : 0,
      matched_symbols: e.matched_symbols ? e.matched_symbols.slice(0, 10) : [],
      matched_tests_count: e.matched_tests ? e.matched_tests.length : 0,
      dependency_evidence_count: e.dependency_evidence ? e.dependency_evidence.length : 0,
      ownership_evidence_count: e.ownership_evidence ? e.ownership_evidence.length : 0,
      negative_evidence: e.negative_evidence || [],
    })),
    recommended_repo: recommendation.recommended_repo,
    confidence: recommendation.confidence,
    explanation: recommendation.explanation,
    scores: recommendation.scores || [],
    timing_ms: totalMs,
  };

  console.log(JSON.stringify(output, null, 2));
}

main().catch(e => {
  console.error(JSON.stringify({
    status: 'ERROR',
    error: e.message,
    stack: e.stack,
  }));
  process.exit(1);
});
