//! Star map data (app/starmap, types StarmapGraph): what Hermes has learned
//! for a profile — its enabled skills as nodes clustered by category and linked
//! by their declared `related_skills`, plus freeform memory as cards.
use serde::{Deserialize, Serialize};
use std::{collections::{BTreeMap, BTreeSet}, path::Path};

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Node { pub id: String, pub label: String, pub category: String, pub description: String }
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Card { pub source: String, pub title: String, pub text: String }
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Graph { pub nodes: Vec<Node>, pub edges: Vec<(String, String)>, pub clusters: Vec<(String, usize)>, pub memory: Vec<Card> }

fn frontmatter<'a>(doc: &'a str, key: &str) -> Option<&'a str> {
    doc.lines().take(40).find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix(':'))).map(|v| v.trim().trim_matches('"'))
}
fn related(doc: &str) -> Vec<String> {
    let mut out = vec![];
    let lines: Vec<&str> = doc.lines().take(60).collect();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.trim_start().strip_prefix("related_skills:") else { continue };
        let rest = rest.trim();
        if rest.starts_with('[') { out.extend(rest.trim_matches(['[', ']']).split(',').map(|s| s.trim().trim_matches(['"', '\'']).to_owned()).filter(|s| !s.is_empty())); }
        else { for next in &lines[i + 1..] { match next.trim_start().strip_prefix("- ") { Some(v) => out.push(v.trim().trim_matches(['"', '\'']).to_owned()), None => break } } }
    }
    out
}

pub fn graph(root: &Path) -> Graph {
    let mut graph = Graph::default();
    if let Ok(status) = crate::skill_catalog::status(root) {
        let mut by_slug = BTreeMap::new();
        let mut docs = vec![];
        for id in status.installed.difference(&status.disabled) {
            let parts: Vec<&str> = id.split('/').collect();
            let slug = parts.last().copied().unwrap_or(id).to_owned();
            let category = parts.iter().rev().nth(1).copied().unwrap_or("general").to_owned();
            let doc = crate::skill_catalog::document(root, id).ok().flatten().unwrap_or_default();
            let description: String = frontmatter(&doc, "description").unwrap_or_default().chars().take(240).collect();
            by_slug.insert(slug.clone(), id.clone());
            graph.nodes.push(Node { id: id.clone(), label: frontmatter(&doc, "name").unwrap_or(&slug).to_owned(), category, description });
            docs.push((id.clone(), doc));
        }
        let mut seen = BTreeSet::new();
        for (id, doc) in &docs {
            for other in related(doc) {
                if let Some(target) = by_slug.get(&other) && target != id {
                    let key = if id < target { (id.clone(), target.clone()) } else { (target.clone(), id.clone()) };
                    if seen.insert(key.clone()) { graph.edges.push(key); }
                }
            }
        }
        let mut clusters = BTreeMap::<String, usize>::new();
        for node in &graph.nodes { *clusters.entry(node.category.clone()).or_default() += 1; }
        graph.clusters = clusters.into_iter().collect();
        graph.clusters.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    }
    for (file, source) in [("MEMORY.md", "memory"), ("USER.md", "profile")] {
        let Ok(text) = std::fs::read_to_string(root.join("memories").join(file)) else { continue };
        for entry in text.split(crate::memory::ENTRY_DELIMITER).map(str::trim).filter(|e| !e.is_empty()) {
            let title: String = entry.lines().next().unwrap_or_default().chars().take(60).collect();
            graph.memory.push(Card { source: source.into(), title, text: entry.chars().take(600).collect() });
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn related_skills_parse_both_yaml_forms() {
        assert_eq!(related("---\nname: a\nrelated_skills: [b, \"c\"]\n---"), ["b", "c"]);
        assert_eq!(related("---\nrelated_skills:\n  - b\n  - c\nother: 1\n---"), ["b", "c"]);
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("memories")).unwrap();
        std::fs::write(dir.path().join("memories/MEMORY.md"), "Likes Rust\n§\nWorks on Hermes").unwrap();
        let g = graph(dir.path());
        assert_eq!(g.memory.len(), 2);
        assert!(!g.nodes.is_empty(), "built-in skills are on the map");
        assert!(g.clusters.iter().map(|c| c.1).sum::<usize>() == g.nodes.len());
    }
}
