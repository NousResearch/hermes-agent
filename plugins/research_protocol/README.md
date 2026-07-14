# Hermes Research Protocol

Ce répertoire contient le lot **PR 0** du protocole de recherche Hermes.

Le plugin est volontairement **opt-in** : son manifeste est découvert sous la
clé `research-protocol`, mais son type `standalone` et l’absence de cette clé
dans `plugins.enabled` empêchent toute activation par défaut. Il ne s’appuie
pas sur un champ de manifeste supplémentaire que le chargeur ignorerait.

PR 0 ne fournit encore aucun outil, handler, skill, profil, stockage ou accès
réseau. Le package est une ancre de découverte et les documents sous `docs/`
gèlent les contrats qui devront précéder toute implémentation runtime.

## Activation

Aucune activation n’est requise ni effectuée par ce lot. Si un environnement
de test l’active explicitement, le point d’entrée reste un no-op : il
n’enregistre aucun outil, hook ou commande. Les capacités runtime ne doivent
être introduites qu’après les tests de leurs contrats de sécurité et une revue
indépendante de la menace.

## Périmètre de sécurité gelé

Les invariants détaillés sont dans [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
et les menaces et contrôles correspondants dans
[`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md). La capacité `publish` reste
manuelle : toute publication externe exige une approbation humaine explicite
portant sur l'opération exacte. Elle ne peut jamais être auto-routée ni
auto-approuvée, y compris sous `/yolo` ou avec `approvals.mode: off`.
