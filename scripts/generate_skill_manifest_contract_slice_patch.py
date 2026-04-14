from pathlib import Path
from difflib import unified_diff

BASE = Path('/home/yeqiuqiu/clawd-architect')
OUT_DIR = BASE / 'patches' / 'openclaw_skill_manifest_contract_slice_2026-04-03' / 'preview'
PATCH_PATH = BASE / 'patches' / 'openclaw_skill_manifest_contract_slice_2026-04-03.patch'


def patch_skills_status(text: str) -> str:
    old = 'import { t as evaluateEntryRequirementsForCurrentPlatform } from "./entry-status-CREA2U_o.js";\nimport path from "node:path";'
    new = 'import { t as evaluateEntryRequirementsForCurrentPlatform } from "./entry-status-CREA2U_o.js";\nimport JSON5 from "json5";\nimport path from "node:path";'
    if old not in text:
        raise RuntimeError('skills-status import block not found')
    text = text.replace(old, new, 1)
    marker = 'function resolveSkillKey(entry) {\n\treturn entry.metadata?.skillKey ?? entry.skill.name;\n}\n'
    insert = '''const SKILL_CONTRACT_VERSION = "openclaw-skill-manifest.v1";
const SAFE_SKILL_KEY_PATTERN = /^[A-Za-z0-9_-]+$/;
const RESERVED_SKILL_KEYS = /* @__PURE__ */ new Set([
\t"__proto__",
\t"prototype",
\t"constructor"
]);
const KNOWN_METADATA_KEYS = /* @__PURE__ */ new Set([
\t"always",
\t"emoji",
\t"homepage",
\t"install",
\t"os",
\t"primaryEnv",
\t"requires",
\t"skillKey"
]);
const KNOWN_REQUIRES_KEYS = /* @__PURE__ */ new Set([
\t"bins",
\t"anyBins",
\t"config",
\t"env"
]);
const KNOWN_INSTALL_KEYS = /* @__PURE__ */ new Set([
\t"archive",
\t"bins",
\t"extract",
\t"formula",
\t"id",
\t"kind",
\t"label",
\t"module",
\t"os",
\t"package",
\t"stripComponents",
\t"targetDir",
\t"type",
\t"url"
]);
const KNOWN_INSTALL_KINDS = /* @__PURE__ */ new Set([
\t"brew",
\t"download",
\t"go",
\t"node",
\t"uv"
]);
function normalizeOptionalString(value) {
\tif (typeof value !== "string") return;
\tconst trimmed = value.trim();
\treturn trimmed || void 0;
}
function normalizeStringList(value) {
\tif (Array.isArray(value)) return value.map((item) => String(item).trim()).filter(Boolean);
\tif (typeof value === "string") return value.split(",").map((item) => item.trim()).filter(Boolean);
\treturn [];
}
function parseExplicitBoolean(value) {
\tif (typeof value !== "string") return { present: false };
\tconst normalized = value.trim().toLowerCase();
\tif (normalized === "true" || normalized === "yes" || normalized === "1") return {
\t\tpresent: true,
\t\tvalid: true,
\t\tvalue: true
\t};
\tif (normalized === "false" || normalized === "no" || normalized === "0") return {
\t\tpresent: true,
\t\tvalid: true,
\t\tvalue: false
\t};
\treturn {
\t\tpresent: true,
\t\tvalid: false,
\t\tvalue: void 0
\t};
}
function addContractIssue(issues, code, level, issuePath, message) {
\tissues.push({
\t\tcode,
\t\tlevel,
\t\tpath: issuePath,
\t\tmessage
\t});
}
function parseRawOpenClawManifest(frontmatter) {
\tconst rawMetadata = normalizeOptionalString(frontmatter?.metadata);
\tif (!rawMetadata) return {
\t\tdeclared: false
\t};
\ttry {
\t\tconst parsed = JSON5.parse(rawMetadata);
\t\tif (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {
\t\t\tdeclared: true,
\t\t\terror: "metadata frontmatter must parse to an object"
\t\t};
\t\tconst manifest = parsed.openclaw;
\t\tif (!manifest || typeof manifest !== "object" || Array.isArray(manifest)) return {
\t\t\tdeclared: true,
\t\t\terror: "metadata frontmatter must contain a metadata.openclaw object"
\t\t};
\t\treturn {
\t\t\tdeclared: true,
\t\t\tmanifest
\t\t};
\t} catch (error) {
\t\tconst detail = error instanceof Error && error.message ? `: ${error.message}` : "";
\t\treturn {
\t\t\tdeclared: true,
\t\t\terror: `metadata frontmatter is not valid JSON5${detail}`
\t\t};
\t}
}
function validateInstallManifestEntry(issues, spec, index) {
\tconst basePath = `metadata.openclaw.install[${index}]`;
\tif (!spec || typeof spec !== "object" || Array.isArray(spec)) {
\t\taddContractIssue(issues, "INVALID_INSTALL_ENTRY", "error", basePath, "install entries must be objects");
\t\treturn {
\t\t\tid: void 0,
\t\t\tvalid: false
\t\t};
\t}
\tfor (const key of Object.keys(spec)) if (!KNOWN_INSTALL_KEYS.has(key)) addContractIssue(issues, "UNKNOWN_INSTALL_KEY", "warn", `${basePath}.${key}`, `unknown install key \"${key}\" will be ignored`);
\tconst kind = typeof spec.kind === "string" ? spec.kind.trim().toLowerCase() : typeof spec.type === "string" ? spec.type.trim().toLowerCase() : "";
\tif (!kind) {
\t\taddContractIssue(issues, "MISSING_INSTALL_KIND", "error", `${basePath}.kind`, "install entries must declare kind or type");
\t\treturn {
\t\t\tid: normalizeOptionalString(spec.id),
\t\t\tvalid: false
\t\t};
\t}
\tif (!KNOWN_INSTALL_KINDS.has(kind)) addContractIssue(issues, "UNSUPPORTED_INSTALL_KIND", "error", `${basePath}.kind`, `unsupported install kind \"${kind}\"`);
\tif (kind == "brew" and not normalizeOptionalString(spec.formula)):
\t\tpass
\treturn {
\t\tid: normalizeOptionalString(spec.id),
\t\tvalid: KNOWN_INSTALL_KINDS.has(kind)
\t};
}
'''
    insert = insert.replace('if (kind == "brew" and not normalizeOptionalString(spec.formula)):\n\t\tpass\n', 'if (kind === "brew" && !normalizeOptionalString(spec.formula)) addContractIssue(issues, "MISSING_INSTALL_FIELD", "error", `${basePath}.formula`, "brew install entries require a formula");\n\tif (kind === "node" && !normalizeOptionalString(spec.package)) addContractIssue(issues, "MISSING_INSTALL_FIELD", "error", `${basePath}.package`, "node install entries require a package");\n\tif (kind === "go" && !normalizeOptionalString(spec.module)) addContractIssue(issues, "MISSING_INSTALL_FIELD", "error", `${basePath}.module`, "go install entries require a module");\n\tif (kind === "uv" && !normalizeOptionalString(spec.package)) addContractIssue(issues, "MISSING_INSTALL_FIELD", "error", `${basePath}.package`, "uv install entries require a package");\n\tif (kind === "download" && !normalizeOptionalString(spec.url)) addContractIssue(issues, "MISSING_INSTALL_FIELD", "error", `${basePath}.url`, "download install entries require a url");\n')
    insert += '''function buildSkillContract(entry) {
\tconst issues = [];
\tconst rawManifest = parseRawOpenClawManifest(entry.frontmatter);
\tconst userInvocableRaw = parseExplicitBoolean(entry.frontmatter?.["user-invocable"]);
\tconst disableModelInvocationRaw = parseExplicitBoolean(entry.frontmatter?.["disable-model-invocation"]);
\tif (userInvocableRaw.present && !userInvocableRaw.valid) addContractIssue(issues, "INVALID_INVOCATION_BOOLEAN", "error", "user-invocable", "user-invocable must be true/false/yes/no/1/0");
\tif (disableModelInvocationRaw.present && !disableModelInvocationRaw.valid) addContractIssue(issues, "INVALID_INVOCATION_BOOLEAN", "error", "disable-model-invocation", "disable-model-invocation must be true/false/yes/no/1/0");
\tif (rawManifest.declared && rawManifest.error) addContractIssue(issues, "INVALID_METADATA_BLOCK", "error", "metadata", rawManifest.error);
\tlet rawInstallCount = 0;
\tif (rawManifest.manifest) {
\t\tfor (const key of Object.keys(rawManifest.manifest)) if (!KNOWN_METADATA_KEYS.has(key)) addContractIssue(issues, "UNKNOWN_METADATA_KEY", "warn", `metadata.openclaw.${key}`, `unknown metadata key \"${key}\" will be ignored`);
\t\tconst skillKey = normalizeOptionalString(rawManifest.manifest.skillKey);
\t\tif (skillKey && !SAFE_SKILL_KEY_PATTERN.test(skillKey)) addContractIssue(issues, "INVALID_SKILL_KEY", "error", "metadata.openclaw.skillKey", `skillKey \"${skillKey}\" must match ${SAFE_SKILL_KEY_PATTERN}`);
\t\tif (skillKey && RESERVED_SKILL_KEYS.has(skillKey)) addContractIssue(issues, "INVALID_SKILL_KEY", "error", "metadata.openclaw.skillKey", `skillKey \"${skillKey}\" is reserved and unsafe`);
\t\tconst requires = rawManifest.manifest.requires;
\t\tlet requiresEnv = [];
\t\tif (requires !== void 0) if (!requires || typeof requires !== "object" || Array.isArray(requires)) addContractIssue(issues, "INVALID_REQUIRES_BLOCK", "error", "metadata.openclaw.requires", "requires must be an object");
\t\telse {
\t\t\tfor (const key of Object.keys(requires)) if (!KNOWN_REQUIRES_KEYS.has(key)) addContractIssue(issues, "UNKNOWN_REQUIRES_KEY", "warn", `metadata.openclaw.requires.${key}`, `unknown requires key \"${key}\" will be ignored`);
\t\t\trequiresEnv = normalizeStringList(requires.env);
\t\t}
\t\tconst primaryEnv = normalizeOptionalString(rawManifest.manifest.primaryEnv);
\t\tif (primaryEnv && requiresEnv.length > 0 && !requiresEnv.includes(primaryEnv)) addContractIssue(issues, "PRIMARY_ENV_NOT_DECLARED", "warn", "metadata.openclaw.primaryEnv", `primaryEnv \"${primaryEnv}\" is not listed in requires.env`);
\t\tconst install = rawManifest.manifest.install;
\t\tif (install !== void 0) if (!Array.isArray(install)) addContractIssue(issues, "INVALID_INSTALL_BLOCK", "error", "metadata.openclaw.install", "install must be an array");
\t\telse {
\t\t\trawInstallCount = install.length;
\t\t\tconst seenInstallIds = /* @__PURE__ */ new Set();
\t\t\tfor (const [index, spec] of install.entries()) {
\t\t\t\tconst validated = validateInstallManifestEntry(issues, spec, index);
\t\t\t\tif (!validated.id) continue;
\t\t\t\tif (seenInstallIds.has(validated.id)) addContractIssue(issues, "DUPLICATE_INSTALL_ID", "warn", `metadata.openclaw.install[${index}].id`, `duplicate install id \"${validated.id}\"`);
\t\t\t\telse seenInstallIds.add(validated.id);
\t\t\t}
\t\t}
\t}
\tconst normalizedInstallCount = entry.metadata?.install?.length ?? 0;
\tif (rawInstallCount > normalizedInstallCount) addContractIssue(issues, "INSTALL_ENTRY_DROPPED", "warn", "metadata.openclaw.install", `${rawInstallCount - normalizedInstallCount} install entr${rawInstallCount - normalizedInstallCount === 1 ? "y was" : "ies were"} ignored during normalization`);
\tconst userInvocable = entry.invocation?.userInvocable !== false;
\tconst disableModelInvocation = entry.invocation?.disableModelInvocation === true;
\tif (!userInvocable && disableModelInvocation) addContractIssue(issues, "NO_INVOCATION_SURFACE", "warn", "invocation", "skill is neither user-invocable nor model-invocable");
\tconst errorCount = issues.filter((issue) => issue.level === "error").length;
\tconst warningCount = issues.filter((issue) => issue.level === "warn").length;
\tconst mode = rawManifest.declared ? "governed" : userInvocableRaw.present || disableModelInvocationRaw.present ? "invocation-only" : "legacy";
\treturn {
\t\tversion: SKILL_CONTRACT_VERSION,
\t\tmode,
\t\tvalid: errorCount === 0,
\t\terrorCount,
\t\twarningCount,
\t\tinvocation: {
\t\t\tuserInvocable,
\t\t\tdisableModelInvocation,
\t\t\tmodelInvocable: !disableModelInvocation
\t\t},
\t\tmanifest: {
\t\t\tmetadataDeclared: rawManifest.declared,
\t\t\tskillKey: entry.metadata?.skillKey ?? entry.skill.name,
\t\t\tprimaryEnv: entry.metadata?.primaryEnv,
\t\t\trawInstallCount,
\t\t\tnormalizedInstallCount
\t\t},
\t\tissues
\t};
}
function resolveSkillKey(entry) {
\treturn entry.metadata?.skillKey ?? entry.skill.name;
}
'''
    if marker not in text:
        raise RuntimeError('skills-status resolveSkillKey marker not found')
    text = text.replace(marker, insert, 1)
    old = '''function buildSkillStatus(entry, config, prefs, eligibility, bundledNames) {
\tconst skillKey = resolveSkillKey(entry);
\tconst skillConfig = resolveSkillConfig(config, skillKey);
\tconst disabled = skillConfig?.enabled === false;
\tconst blockedByAllowlist = !isBundledSkillAllowed(entry, resolveBundledAllowlist(config));
\tconst always = entry.metadata?.always === true;
\tconst isEnvSatisfied = (envName) => Boolean(process.env[envName] || skillConfig?.env?.[envName] || skillConfig?.apiKey && entry.metadata?.primaryEnv === envName);
\tconst isConfigSatisfied = (pathStr) => isConfigPathTruthy(config, pathStr);
\tconst skillSource = resolveSkillSource(entry.skill);
\tconst bundled = skillSource === "openclaw-bundled" || skillSource === "unknown" && bundledNames?.has(entry.skill.name) === true;
\tconst { emoji, homepage, required, missing, requirementsSatisfied, configChecks } = evaluateEntryRequirementsForCurrentPlatform({
\t\talways,
\t\tentry,
\t\thasLocalBin: hasBinary,
\t\tremote: eligibility?.remote,
\t\tisEnvSatisfied,
\t\tisConfigSatisfied
\t});
\tconst eligible = !disabled && !blockedByAllowlist && requirementsSatisfied;
\treturn {
\t\tname: entry.skill.name,
\t\tdescription: entry.skill.description,
\t\tsource: skillSource,
\t\tbundled,
\t\tfilePath: entry.skill.filePath,
\t\tbaseDir: entry.skill.baseDir,
\t\tskillKey,
\t\tprimaryEnv: entry.metadata?.primaryEnv,
\t\temoji,
\t\thomepage,
\t\talways,
\t\tdisabled,
\t\tblockedByAllowlist,
\t\teligible,
\t\trequirements: required,
\t\tmissing,
\t\tconfigChecks,
\t\tinstall: normalizeInstallOptions(entry, prefs ?? resolveSkillsInstallPreferences(config))
\t};
}
'''
    new = '''function buildSkillStatus(entry, config, prefs, eligibility, bundledNames) {
\tconst skillKey = resolveSkillKey(entry);
\tconst contract = buildSkillContract(entry);
\tconst skillConfig = resolveSkillConfig(config, skillKey);
\tconst disabled = skillConfig?.enabled === false;
\tconst blockedByAllowlist = !isBundledSkillAllowed(entry, resolveBundledAllowlist(config));
\tconst always = entry.metadata?.always === true;
\tconst isEnvSatisfied = (envName) => Boolean(process.env[envName] || skillConfig?.env?.[envName] || skillConfig?.apiKey && entry.metadata?.primaryEnv === envName);
\tconst isConfigSatisfied = (pathStr) => isConfigPathTruthy(config, pathStr);
\tconst skillSource = resolveSkillSource(entry.skill);
\tconst bundled = skillSource === "openclaw-bundled" || skillSource === "unknown" && bundledNames?.has(entry.skill.name) === true;
\tconst { emoji, homepage: resolvedHomepage, required, missing, requirementsSatisfied, configChecks } = evaluateEntryRequirementsForCurrentPlatform({
\t\talways,
\t\tentry,
\t\thasLocalBin: hasBinary,
\t\tremote: eligibility?.remote,
\t\tisEnvSatisfied,
\t\tisConfigSatisfied
\t});
\tconst homepage = resolvedHomepage ?? normalizeOptionalString(entry.frontmatter?.homepage);
\tconst eligible = !disabled && !blockedByAllowlist && requirementsSatisfied;
\treturn {
\t\tname: entry.skill.name,
\t\tdescription: entry.skill.description,
\t\tsource: skillSource,
\t\tbundled,
\t\tfilePath: entry.skill.filePath,
\t\tbaseDir: entry.skill.baseDir,
\t\tskillKey,
\t\tprimaryEnv: entry.metadata?.primaryEnv,
\t\temoji,
\t\thomepage,
\t\tcontract,
\t\tcontractIssues: contract.issues,
\t\talways,
\t\tdisabled,
\t\tblockedByAllowlist,
\t\teligible,
\t\trequirements: required,
\t\tmissing,
\t\tconfigChecks,
\t\tinstall: normalizeInstallOptions(entry, prefs ?? resolveSkillsInstallPreferences(config))
\t};
}
'''
    if old not in text:
        raise RuntimeError('skills-status buildSkillStatus block not found')
    return text.replace(old, new, 1)


def patch_skills_cli(text: str) -> str:
    marker = 'function formatSkillMissingSummary(skill) {\n\tconst missing = [];\n\tif (skill.missing.bins.length > 0) missing.push(`bins: ${skill.missing.bins.join(", ")}`);\n\tif (skill.missing.anyBins.length > 0) missing.push(`anyBins: ${skill.missing.anyBins.join(", ")}`);\n\tif (skill.missing.env.length > 0) missing.push(`env: ${skill.missing.env.join(", ")}`);\n\tif (skill.missing.config.length > 0) missing.push(`config: ${skill.missing.config.join(", ")}`);\n\tif (skill.missing.os.length > 0) missing.push(`os: ${skill.missing.os.join(", ")}`);\n\treturn missing.join("; ");\n}\n'
    insert = '''function formatSkillMissingSummary(skill) {
\tconst missing = [];
\tif (skill.missing.bins.length > 0) missing.push(`bins: ${skill.missing.bins.join(", ")}`);
\tif (skill.missing.anyBins.length > 0) missing.push(`anyBins: ${skill.missing.anyBins.join(", ")}`);
\tif (skill.missing.env.length > 0) missing.push(`env: ${skill.missing.env.join(", ")}`);
\tif (skill.missing.config.length > 0) missing.push(`config: ${skill.missing.config.join(", ")}`);
\tif (skill.missing.os.length > 0) missing.push(`os: ${skill.missing.os.join(", ")}`);
\treturn missing.join("; ");
}
function summarizeSkillContract(skill) {
\tif (!skill.contract) return;
\treturn {
\t\tversion: skill.contract.version,
\t\tmode: skill.contract.mode,
\t\tvalid: skill.contract.valid,
\t\terrorCount: skill.contract.errorCount,
\t\twarningCount: skill.contract.warningCount
\t};
}
function formatContractIssue(issue) {
\tconst pathSuffix = issue.path ? ` @ ${sanitizeForLog(issue.path)}` : "";
\treturn `${sanitizeForLog(issue.code)}${pathSuffix} — ${sanitizeForLog(issue.message)}`;
}
function collectContractFindings(skills) {
\tconst findings = [];
\tlet errors = 0;
\tlet warnings = 0;
\tfor (const skill of skills) {
\t\tconst issues = skill.contract?.issues ?? [];
\t\tif (issues.length === 0) continue;
\t\tfor (const issue of issues) if (issue.level === "error") errors += 1;
\t\telse warnings += 1;
\t\tfindings.push({
\t\t\tname: skill.name,
\t\t\temoji: skill.emoji,
\t\t\tskillKey: skill.skillKey,
\t\t\tissues
\t\t});
\t}
\treturn {
\t\terrors,
\t\twarnings,
\t\tfindings
\t};
}
'''
    if marker not in text:
        raise RuntimeError('skills-cli marker not found')
    text = text.replace(marker, insert, 1)
    old = '''\t\t\tskills: skills.map((s) => ({
\t\t\t\tname: s.name,
\t\t\t\tdescription: s.description,
\t\t\t\temoji: s.emoji,
\t\t\t\teligible: s.eligible,
\t\t\t\tdisabled: s.disabled,
\t\t\t\tblockedByAllowlist: s.blockedByAllowlist,
\t\t\t\tsource: s.source,
\t\t\t\tbundled: s.bundled,
\t\t\t\tprimaryEnv: s.primaryEnv,
\t\t\t\thomepage: s.homepage,
\t\t\t\tmissing: s.missing
\t\t\t}))
'''
    new = '''\t\t\tskills: skills.map((s) => ({
\t\t\t\tname: s.name,
\t\t\t\tdescription: s.description,
\t\t\t\temoji: s.emoji,
\t\t\t\teligible: s.eligible,
\t\t\t\tdisabled: s.disabled,
\t\t\t\tblockedByAllowlist: s.blockedByAllowlist,
\t\t\t\tsource: s.source,
\t\t\t\tbundled: s.bundled,
\t\t\t\tprimaryEnv: s.primaryEnv,
\t\t\t\thomepage: s.homepage,
\t\t\t\tmissing: s.missing,
\t\t\t\tcontract: summarizeSkillContract(s)
\t\t\t}))
'''
    if old not in text:
        raise RuntimeError('skills-cli list json block not found')
    text = text.replace(old, new, 1)
    old = 'if (skill.primaryEnv) lines.push(`${theme.muted("  Primary env:")} ${skill.primaryEnv}`);\n\tif (skill.requirements.bins.length > 0 || skill.requirements.anyBins.length > 0 || skill.requirements.env.length > 0 || skill.requirements.config.length > 0 || skill.requirements.os.length > 0) {'
    new = '''if (skill.primaryEnv) lines.push(`${theme.muted("  Primary env:")} ${skill.primaryEnv}`);
\tif (skill.contract) {
\t\tlines.push("");
\t\tlines.push(theme.heading("Contract:"));
\t\tlines.push(`${theme.muted("  Version:")} ${sanitizeForLog(skill.contract.version)} ${theme.muted(`(${sanitizeForLog(skill.contract.mode)})`)} ${skill.contract.valid ? theme.success("✓ valid") : theme.error("✗ invalid")}`);
\t\tlines.push(`${theme.muted("  Invocation:")} ${skill.contract.invocation.userInvocable ? "user" : "not-user"}, ${skill.contract.invocation.modelInvocable ? "model" : "not-model"}`);
\t\tif (skill.contract.issues.length > 0) {
\t\t\tlines.push(`${theme.muted("  Findings:")} ${skill.contract.issues.length}`);
\t\t\tfor (const issue of skill.contract.issues) lines.push(`    ${issue.level === "error" ? theme.error("✗") : theme.warn("△")} ${formatContractIssue(issue)}`);
\t\t}
\t}
\tif (skill.requirements.bins.length > 0 || skill.requirements.anyBins.length > 0 || skill.requirements.env.length > 0 || skill.requirements.config.length > 0 || skill.requirements.os.length > 0) {'''
    if old not in text:
        raise RuntimeError('skills-cli info insertion point not found')
    text = text.replace(old, new, 1)
    old = 'const missingReqs = report.skills.filter((s) => !s.eligible && !s.disabled && !s.blockedByAllowlist);\n\tif (opts.json) return JSON.stringify(sanitizeJsonValue({'
    new = 'const missingReqs = report.skills.filter((s) => !s.eligible && !s.disabled && !s.blockedByAllowlist);\n\tconst contractFindings = collectContractFindings(report.skills);\n\tif (opts.json) return JSON.stringify(sanitizeJsonValue({'
    if old not in text:
        raise RuntimeError('skills-cli check header block not found')
    text = text.replace(old, new, 1)
    old = '''\t\tsummary: {
\t\t\ttotal: report.skills.length,
\t\t\teligible: eligible.length,
\t\t\tdisabled: disabled.length,
\t\t\tblocked: blocked.length,
\t\t\tmissingRequirements: missingReqs.length
\t\t},
\t\teligible: eligible.map((s) => s.name),
\t\tdisabled: disabled.map((s) => s.name),
\t\tblocked: blocked.map((s) => s.name),
\t\tmissingRequirements: missingReqs.map((s) => ({
\t\t\tname: s.name,
\t\t\tmissing: s.missing,
\t\t\tinstall: s.install
\t\t}))
\t}), null, 2);'''
    new = '''\t\tsummary: {
\t\t\ttotal: report.skills.length,
\t\t\teligible: eligible.length,
\t\t\tdisabled: disabled.length,
\t\t\tblocked: blocked.length,
\t\t\tmissingRequirements: missingReqs.length,
\t\t\tcontractErrors: contractFindings.errors,
\t\t\tcontractWarnings: contractFindings.warnings,
\t\t\tskillsWithContractFindings: contractFindings.findings.length
\t\t},
\t\teligible: eligible.map((s) => s.name),
\t\tdisabled: disabled.map((s) => s.name),
\t\tblocked: blocked.map((s) => s.name),
\t\tmissingRequirements: missingReqs.map((s) => ({
\t\t\tname: s.name,
\t\t\tmissing: s.missing,
\t\t\tinstall: s.install
\t\t})),
\t\tcontractFindings: contractFindings.findings.map((skill) => ({
\t\t\tname: skill.name,
\t\t\tskillKey: skill.skillKey,
\t\t\tissues: skill.issues
\t\t}))
\t}), null, 2);'''
    if old not in text:
        raise RuntimeError('skills-cli check json block not found')
    text = text.replace(old, new, 1)
    old = 'if (missingReqs.length > 0) {\n\t\tlines.push("");\n\t\tlines.push(theme.heading("Missing requirements:"));\n\t\tfor (const skill of missingReqs) {\n\t\t\tconst emoji = normalizeSkillEmoji(skill.emoji);\n\t\t\tconst missing = formatSkillMissingSummary(skill);\n\t\t\tlines.push(`  ${emoji} ${sanitizeForLog(skill.name)} ${theme.muted(`(${missing})`)}`);\n\t\t}\n\t}\n\treturn appendClawHubHint(lines.join("\\n"), opts.json);'
    new = '''if (missingReqs.length > 0) {
\t\tlines.push("");
\t\tlines.push(theme.heading("Missing requirements:"));
\t\tfor (const skill of missingReqs) {
\t\t\tconst emoji = normalizeSkillEmoji(skill.emoji);
\t\t\tconst missing = formatSkillMissingSummary(skill);
\t\t\tlines.push(`  ${emoji} ${sanitizeForLog(skill.name)} ${theme.muted(`(${missing})`)}`);
\t\t}
\t}
\tif (contractFindings.findings.length > 0) {
\t\tlines.push("");
\t\tlines.push(theme.heading("Contract findings:"));
\t\tlines.push(`  ${theme.error("✗")} ${theme.muted("Errors:")} ${contractFindings.errors}`);
\t\tlines.push(`  ${theme.warn("△")} ${theme.muted("Warnings:")} ${contractFindings.warnings}`);
\t\tfor (const skill of contractFindings.findings) {
\t\t\tconst emoji = normalizeSkillEmoji(skill.emoji);
\t\t\tlines.push(`  ${emoji} ${sanitizeForLog(skill.name)}`);
\t\t\tfor (const issue of skill.issues) lines.push(`    ${issue.level === "error" ? theme.error("✗") : theme.warn("△")} ${formatContractIssue(issue)}`);
\t\t}
\t}
\treturn appendClawHubHint(lines.join("\\n"), opts.json);'''
    if old not in text:
        raise RuntimeError('skills-cli check footer block not found')
    return text.replace(old, new, 1)


def patch_spotify(text: str) -> str:
    old = '''            {
              "id": "brew",
              "kind": "brew",
              "formula": "spogo",
              "tap": "steipete/tap",
              "bins": ["spogo"],
              "label": "Install spogo (brew)",
            },
            {
              "id": "brew",
              "kind": "brew",
              "formula": "spotify_player",
'''
    new = '''            {
              "id": "brew-spogo",
              "kind": "brew",
              "formula": "steipete/tap/spogo",
              "bins": ["spogo"],
              "label": "Install spogo (brew)",
            },
            {
              "id": "brew-spotify-player",
              "kind": "brew",
              "formula": "spotify_player",
'''
    if old not in text:
        raise RuntimeError('spotify-player install block not found')
    return text.replace(old, new, 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        Path('/usr/lib/node_modules/openclaw/dist/skills-status-6dqJ2gft.js'),
        Path('/usr/lib/node_modules/openclaw/dist/skills-cli-GuSPe1vI.js'),
        Path('/usr/lib/node_modules/openclaw/skills/spotify-player/SKILL.md'),
    ]
    modifiers = {
        str(targets[0]): patch_skills_status,
        str(targets[1]): patch_skills_cli,
        str(targets[2]): patch_spotify,
    }
    all_diffs = []
    for orig in targets:
        old_text = orig.read_text()
        new_text = modifiers[str(orig)](old_text)
        preview = OUT_DIR / orig.relative_to('/')
        preview.parent.mkdir(parents=True, exist_ok=True)
        preview.write_text(new_text)
        all_diffs.extend(unified_diff(
            old_text.splitlines(True),
            new_text.splitlines(True),
            fromfile=str(orig),
            tofile=str(orig),
        ))
    PATCH_PATH.write_text(''.join(all_diffs))
    print(PATCH_PATH)


if __name__ == '__main__':
    main()
