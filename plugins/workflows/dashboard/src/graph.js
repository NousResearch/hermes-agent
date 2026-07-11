// Pure adapter for spec → graph membership. Triggers get rendererType "trigger" so
// they stay distinct from nodes whose rendererType equals their node.type; this
// keeps React Flow identities stable across renders and decouples renderer
// identity from persisted trigger.type.

export function graphItems(spec) {
  const triggers = (spec?.triggers || []).map((trigger, index) => ({
    id: trigger.id || trigger.name || `trigger_${index + 1}`,
    rendererType: "trigger",
    specKind: "trigger",
    triggerType: trigger.type,
    spec: trigger,
  }));
  const nodes = Object.entries(spec?.nodes || {}).map(([id, node]) => ({
    id,
    rendererType: node.type,
    specKind: "node",
    triggerType: null,
    spec: node,
  }));
  return [...triggers, ...nodes];
}

export function decorateGraphItems(items, statuses) {
  return items.map((item) => ({ ...item, status: statuses[item.id] || "idle" }));
}
