#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const options = parseArgs(process.argv.slice(2));

if (options.help || !options.name) {
  printUsage();
  process.exit(options.help ? 0 : 1);
}

const slug = kebab(options.slug || options.name);
const componentName = pascal(`${slug}-page`);
const title = options.title || titleCase(slug);
const route = normalizeRoute(options.route || `/${slug}`);
const recipe = resolveRecipe(options.recipe);
const category = options.category || recipe?.category || "operations";
const owner = options.owner || "Hermes";
const manifestPath = path.resolve(root, options.manifest || "hermes.dashboards.json");
const pagePath = path.resolve(root, options.page || `web/src/pages/${componentName}.tsx`);
const testPath = path.resolve(root, options.test || `tests/dashboard/${slug}.spec.ts`);
const routeRegistryPath = path.resolve(root, options.routeRegistry || options.app || "web/src/dashboard-route-registry.tsx");
const registerRoute = options.register !== "false";

ensureInside(root, pagePath);
ensureInside(root, testPath);
ensureInside(root, manifestPath);
ensureInside(root, routeRegistryPath);

writeNewFile(pagePath, recipe ? recipePageTemplate({ componentName, title, recipe }) : pageTemplate({ componentName, title }));
writeNewFile(testPath, testTemplate({ route, title }));
upsertManifest(manifestPath, {
  id: options.id || `nous-hermes-agent.${slug}`,
  label: title,
  description: options.description || `${title} dashboard scaffolded from the Hermes dashboard kit.`,
  category,
  owner,
  url: route,
  command: [],
});

if (registerRoute) {
  registerRouteRegistry(routeRegistryPath, { componentName, route, title });
}

console.log(`Created dashboard scaffold: ${title}`);
if (recipe) console.log(`Recipe: ${recipe.title}`);
console.log(`Page: ${path.relative(root, pagePath)}`);
console.log(`Test: ${path.relative(root, testPath)}`);
console.log(`Manifest: ${path.relative(root, manifestPath)}`);
if (registerRoute) console.log(`Route: ${route}`);

function parseArgs(args) {
  const parsed = {};
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--help" || arg === "-h") {
      parsed.help = true;
      continue;
    }
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected argument: ${arg}`);
    }
    const key = arg.slice(2);
    const next = args[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = "true";
      continue;
    }
    parsed[key] = next;
    index += 1;
  }
  return parsed;
}

function printUsage() {
  console.log(`Usage:
  node scripts/scaffold-dashboard-page.mjs --name "Operations Intelligence"

Options:
  --name           Required dashboard name.
  --slug           Route/file slug. Defaults from name.
  --title          Display title. Defaults from slug.
  --route          Route path. Defaults to /<slug>.
  --category       Manifest category. Defaults to operations.
  --owner          Manifest owner. Defaults to Hermes.
  --description    Manifest description.
  --recipe         Approved V7 recipe id. Use --recipe list to print options.
  --id             Manifest id.
  --manifest       Manifest path. Defaults to hermes.dashboards.json.
  --page           Page output path.
  --test           Playwright test output path.
  --routeRegistry  Route registry file. Defaults to web/src/dashboard-route-registry.tsx.
  --app            Deprecated alias for --routeRegistry.
  --register       Set false to skip route registration.`);
}

function getDashboardRecipes() {
  return [
  {
    id: "executive-command-center",
    title: "Executive Command Center",
    category: "executive",
    sections: ["Portfolio Health", "Domain Scorecards", "Action Needed", "Project Launchers", "Executive Recommendations"],
    metrics: ["Portfolio Health", "Open Actions", "Capacity Risk"],
    tableTitle: "Action Queue",
    chartTitle: "Domain Throughput",
    sidebar: ["Overview", "Domains", "Actions", "Projects", "Recommendations"],
    contract: ["projects[]", "actions[]", "domains[]"],
  },
  {
    id: "operations-control-room",
    title: "Operations Control Room",
    category: "operations",
    sections: ["Command Center", "Capacity", "Run Monitor", "Worker Health", "Audit Trail"],
    metrics: ["Running", "Queued", "Available Capacity"],
    tableTitle: "Runs",
    chartTitle: "Queue Pressure",
    sidebar: ["Overview", "Commands", "Runs", "Workers", "Audit"],
    contract: ["commands[]", "capacity", "runs[]", "workers[]"],
  },
  {
    id: "research-intelligence-dashboard",
    title: "Research Intelligence Dashboard",
    category: "research",
    sections: ["Evidence Summary", "Findings", "Coverage Heatmap", "Experiment Results", "Next Research Action"],
    metrics: ["Evidence", "Findings", "Blind Spots"],
    tableTitle: "Experiments",
    chartTitle: "Coverage",
    sidebar: ["Overview", "Evidence", "Findings", "Coverage", "Experiments"],
    contract: ["findings[]", "coverage[]", "experiments[]"],
  },
  {
    id: "pipeline-workflow-dashboard",
    title: "Pipeline Workflow Dashboard",
    category: "media-operations",
    sections: ["Pipeline Health", "Brand Jobs", "Approval Queue", "Discord Output", "Failure Review"],
    metrics: ["Due Jobs", "Delivered", "Failed"],
    tableTitle: "Pipeline Jobs",
    chartTitle: "Stage Distribution",
    sidebar: ["Overview", "Jobs", "Approvals", "Output", "Failures"],
    contract: ["jobs[]", "approvals[]", "outputs[]"],
  },
  {
    id: "cost-capacity-dashboard",
    title: "Cost And Capacity Dashboard",
    category: "operations-intelligence",
    sections: ["Budget Posture", "Usage Trends", "Resource Breakdown", "Budget Risk", "Recommendations"],
    metrics: ["Daily Spend", "API Calls", "Storage Used"],
    tableTitle: "Usage Breakdown",
    chartTitle: "Cost Trend",
    sidebar: ["Overview", "Budget", "Usage", "Resources", "Risk"],
    contract: ["usageSeries[]", "budgets[]", "resources[]"],
  },
  {
    id: "market-asset-explorer",
    title: "Market Asset Explorer",
    category: "research-operations",
    sections: ["Universe Summary", "Filters", "Category Tag Heatmap", "Market Results", "Selected Asset"],
    metrics: ["Live Assets", "Tags Covered", "Closing Soon"],
    tableTitle: "Assets",
    chartTitle: "Category Coverage",
    sidebar: ["Overview", "Filters", "Heatmap", "Assets", "Detail"],
    contract: ["assets[]", "coverage[]", "selectedAsset"],
  },
  {
    id: "brand-business-performance",
    title: "Brand Business Performance Dashboard",
    category: "media-operations",
    sections: ["Brand Health", "Output Cadence", "Channel Performance", "Content Calendar", "Recommendations"],
    metrics: ["Active Brands", "Posts Due", "Missed Cadence"],
    tableTitle: "Content Cadence",
    chartTitle: "Channel Trend",
    sidebar: ["Overview", "Brands", "Cadence", "Channels", "Content"],
    contract: ["brands[]", "channels[]", "content[]"],
  },
  {
    id: "system-health-deployment",
    title: "System Health And Deployment Dashboard",
    category: "system",
    sections: ["Service Health", "Deployment Timeline", "CI Checks", "Manifest Registry", "Release Actions"],
    metrics: ["Healthy Services", "Failed Checks", "Deploy Age"],
    tableTitle: "Checks",
    chartTitle: "Service Health",
    sidebar: ["Overview", "Services", "Deploys", "Checks", "Registry"],
    contract: ["services[]", "deployments[]", "checks[]", "secrets[]"],
  },
  ];
}

function resolveRecipe(value) {
  if (!value) return null;
  if (value === "list") {
    printRecipes();
    process.exit(0);
  }
  const normalized = kebab(value);
  const recipe = getDashboardRecipes().find((entry) => entry.id === normalized);
  if (!recipe) {
    printRecipes();
    throw new Error(`Unknown dashboard recipe: ${value}`);
  }
  return recipe;
}

function printRecipes() {
  console.log("Approved V7 dashboard recipes:");
  for (const recipe of getDashboardRecipes()) {
    console.log(`- ${recipe.id}: ${recipe.title}`);
  }
}

function kebab(value) {
  return String(value)
    .trim()
    .replace(/['"]/g, "")
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

function pascal(value) {
  return kebab(value)
    .split("-")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join("");
}

function titleCase(value) {
  return kebab(value)
    .split("-")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function normalizeRoute(value) {
  const route = String(value || "").trim();
  return route.startsWith("/") ? route : `/${route}`;
}

function ensureInside(base, target) {
  const relative = path.relative(base, target);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`Refusing to write outside repo: ${target}`);
  }
}

function writeNewFile(file, content) {
  if (fs.existsSync(file)) {
    throw new Error(`Refusing to overwrite existing file: ${file}`);
  }
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
}

function upsertManifest(file, dashboard) {
  const registry = fs.existsSync(file)
    ? JSON.parse(fs.readFileSync(file, "utf8"))
    : { version: 1, dashboards: [] };
  if (!Array.isArray(registry.dashboards)) {
    throw new Error(`${file} does not contain dashboards[]`);
  }
  const existing = registry.dashboards.find((entry) => entry.id === dashboard.id);
  if (existing) Object.assign(existing, dashboard);
  else registry.dashboards.push(dashboard);
  fs.writeFileSync(file, `${JSON.stringify(registry, null, 2)}\n`);
}

function registerRouteRegistry(file, dashboard) {
  if (!fs.existsSync(file)) throw new Error(`Route registry file not found: ${file}`);
  let text = fs.readFileSync(file, "utf8");
  if (text.includes(`"${dashboard.route}"`)) return;
  const routeLine = `const ${dashboard.componentName} = lazy(() => import("@/pages/${dashboard.componentName}"));\n`;
  text = text.replace(
    /const DesignSystemPage = lazy\(\(\) => import\("@\/pages\/DesignSystemPage"\)\);\n/,
    (match) => `${match}${routeLine}`,
  );
  text = text.replace(
    /  "\/design-system": DesignSystemPage,\n/,
    (match) => `  "${dashboard.route}": ${dashboard.componentName},\n${match}`,
  );
  text = text.replace(
    /  \{ path: "\/design-system", label: "Design System", icon: GalleryVerticalEnd \},\n/,
    (match) => `  { path: "${dashboard.route}", label: "${dashboard.title}", icon: GalleryVerticalEnd },\n${match}`,
  );
  if (!text.includes(`"${dashboard.route}"`)) {
    throw new Error(`Could not register ${dashboard.route} in ${file}; registry anchors changed`);
  }
  fs.writeFileSync(file, text);
}

function pageTemplate({ componentName, title }) {
  return `import { BarChart3, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  ChartPanel,
  DashboardEmptyState,
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  KpiCard,
  MetricGrid,
  SimpleBarChart,
  StatusPill,
  type DataTableColumn,
} from "@hermes/dashboard-kit";

interface Row {
  id: string;
  status: string;
  owner: string;
}

const rows: Row[] = [
  { id: "item-1", status: "ready", owner: "Hermes" },
  { id: "item-2", status: "watch", owner: "Operations" },
];

const columns: DataTableColumn<Row>[] = [
  { id: "id", header: "ID", accessor: (row) => row.id, sortValue: (row) => row.id },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone={row.status === "ready" ? "success" : "warning"}>{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
];

export default function ${componentName}() {
  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="${escapeTemplate(title)}"
          description="Scaffolded dashboard."
          items={[
            { id: "overview", label: "Overview", href: "#overview", active: true, icon: BarChart3 },
            { id: "records", label: "Records", href: "#records" },
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="${escapeTemplate(title)}"
          eyebrow="Dashboard scaffold"
          description="Replace sample data with real sources after completing the dashboard planning template."
          actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>}
        />
      )}
    >
      <MetricGrid id="overview" columns={3}>
        <KpiCard label="Health" value="Ready" tone="success" detail="Scaffold rendered" />
        <KpiCard label="Rows" value={rows.length} detail="Sample records" />
        <KpiCard label="Actions" value="0" detail="No operator actions yet" />
      </MetricGrid>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
        <DashboardSection id="records" title="Records" description="Replace with the primary dashboard record set.">
          <DataTable columns={columns} rows={rows} getRowKey={(row) => row.id} />
        </DashboardSection>
        <DashboardSection title="Trend" description="Replace with a meaningful operational series.">
          <ChartPanel title="Sample Chart">
            <SimpleBarChart data={[{ label: "A", value: 4 }, { label: "B", value: 7 }]} />
          </ChartPanel>
        </DashboardSection>
      </div>

      <DashboardEmptyState title="Add real data" description="Wire this dashboard to project APIs or files before production use." />
    </DashboardShell>
  );
}
`;
}

function recipePageTemplate({ componentName, title, recipe }) {
  const sidebarItems = recipe.sidebar
    .map((label, index) => {
      const id = kebab(label);
      const icon = index === 0 ? ", icon: BarChart3" : "";
      return `{ id: "${id}", label: "${escapeTemplate(label)}", href: "#${id}"${index === 0 ? ", active: true" : ""}${icon} }`;
    })
    .join(",\n            ");
  const metricCards = recipe.metrics
    .map((label, index) => `<KpiCard label="${escapeTemplate(label)}" value="${index === 0 ? "Ready" : "0"}" detail="Replace with ${escapeTemplate(recipe.id)} data." tone="${index === 0 ? "info" : "neutral"}" />`)
    .join("\n        ");
  const contractItems = recipe.contract.map((item) => `<li>${escapeTemplate(item)}</li>`).join("\n              ");
  const sectionItems = recipe.sections.map((item) => `<li>${escapeTemplate(item)}</li>`).join("\n              ");
  const tableRows = recipe.contract.map((item, index) => `{ id: "contract-${index + 1}", name: "${escapeTemplate(item)}", status: "required", owner: "${escapeTemplate(recipe.title)}" }`).join(",\n  ");

  return `import { BarChart3, RefreshCw, ShieldCheck } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  ChartPanel,
  CommandBar,
  DashboardEmptyState,
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  KpiCard,
  MetricGrid,
  SimpleBarChart,
  StatusPill,
  type DataTableColumn,
} from "@hermes/dashboard-kit";

interface ContractRow {
  id: string;
  name: string;
  status: string;
  owner: string;
}

const rows: ContractRow[] = [
  ${tableRows}
];

const columns: DataTableColumn<ContractRow>[] = [
  { id: "name", header: "Required Data", accessor: (row) => row.name, sortValue: (row) => row.name },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone="warning">{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "owner", header: "Recipe", accessor: (row) => row.owner, sortValue: (row) => row.owner },
];

export default function ${componentName}() {
  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="${escapeTemplate(title)}"
          description="${escapeTemplate(recipe.title)} recipe."
          items={[
            ${sidebarItems}
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="${escapeTemplate(title)}"
          eyebrow="V7 ${escapeTemplate(recipe.title)}"
          description="Approved full-page dashboard recipe scaffold. Replace sample content with live project data before production use."
          actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>}
          meta={<StatusPill tone="info">${escapeTemplate(recipe.id)}</StatusPill>}
        />
      )}
    >
      <MetricGrid id="overview" columns={3}>
        ${metricCards}
      </MetricGrid>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <DashboardSection id="${kebab(recipe.sidebar[1] || "contract")}" title="Recipe Contract" description="Wire these data groups before marking the dashboard production-ready.">
          <DataTable columns={columns} rows={rows} getRowKey={(row) => row.id} />
        </DashboardSection>
        <DashboardSection title="Required Sections" description="The page should preserve this operating anatomy.">
          <ul className="space-y-2 text-sm text-muted-foreground">
            ${sectionItems}
          </ul>
        </DashboardSection>
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <DashboardSection id="${kebab(recipe.sidebar[2] || "commands")}" title="${escapeTemplate(recipe.tableTitle)}" description="Replace placeholder rows with live records.">
          <DataTable columns={columns} rows={rows} getRowKey={(row) => row.id} />
        </DashboardSection>
        <ChartPanel title="${escapeTemplate(recipe.chartTitle)}" description="Replace sample values with the recipe's primary trend or distribution.">
          <SimpleBarChart data={[{ label: "A", value: 4 }, { label: "B", value: 7 }, { label: "C", value: 5 }]} />
        </ChartPanel>
      </div>

      <CommandBar
        title="Recipe Actions"
        description="Operator actions must declare permission, disabled reason, and risk before production use."
        actions={[
          { id: "verify", label: "Verify Data", icon: ShieldCheck, permission: "operator", riskLevel: "low" },
          { id: "promote", label: "Promote", disabled: true, disabledReason: "Connect live data and Playwright coverage first.", permission: "admin", riskLevel: "high" },
        ]}
      />

      <DashboardEmptyState title="Finish the ${escapeTemplate(recipe.title)} wiring" description="Complete the required data contract, states, and validation before this dashboard becomes an operating source of truth." />
    </DashboardShell>
  );
}
`;
}

function testTemplate({ route, title }) {
  return `import { expect, test } from "@playwright/test";

test("${escapeTemplate(title)} dashboard renders", async ({ page }) => {
  await page.goto("${route}");
  await expect(page.getByRole("heading", { name: /${escapeRegExp(title)}/i }).first()).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(2);
});
`;
}

function escapeTemplate(value) {
  return String(value).replace(/\\/g, "\\\\").replace(/`/g, "\\`").replace(/\$/g, "\\$");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
