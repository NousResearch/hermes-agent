#!/usr/bin/env bash
# Script pour exécuter les tests du skill security-recon-assistant

set -euo pipefail

# Couleurs pour l'affichage
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Répertoire du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Security Recon Assistant - Test Suite${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

# Vérifier si on est dans un venv Python
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo -e "${YELLOW}⚠️  Attention: Aucun environnement virtuel Python activé${NC}"
    echo "   Il est recommandé d'utiliser un venv:"
    echo "   python -m venv venv && source venv/bin/activate"
    echo ""
    read -p "Continuer quand même? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Installer les dépendances de développement si nécessaire
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${YELLOW}📦 Installation des dépendances de test...${NC}"
    pip install -e ".[dev]"
fi

# Fonction pour exécuter une catégorie de tests
run_tests() {
    local category=$1
    local marker=$2
    echo ""
    echo -e "${BLUE}▶ $category${NC}"
    set +e
    pytest -m "$marker" -v --tb=short
    local status=$?
    set -e

    if [[ $status -eq 0 ]]; then
        echo -e "${GREEN}✓ $category: OK${NC}"
        return 0
    fi

    # pytest retourne 5 si aucun test ne correspond au marqueur.
    # Dans ce cas, fallback sur l'ensemble des tests pour rester opérable.
    if [[ $status -eq 5 ]]; then
        echo -e "${YELLOW}⚠ Aucun test marqué '$marker' détecté, fallback sur toute la suite.${NC}"
        if pytest -v --tb=short; then
            echo -e "${GREEN}✓ $category (fallback): OK${NC}"
            return 0
        fi
    fi

    echo -e "${RED}✗ $category: FAILED${NC}"
    return 1
}

# Variables pour suivre le statut global
overall_status=0

# 1. Tests rapides (unitaires)
run_tests "Tests unitaires" "unit" || overall_status=1

# 2. Tests d'intégration
run_tests "Tests d'intégration" "integration" || overall_status=1

# 3. Optionnel: Tests lents (désactivés par défaut)
if [[ "${RUN_SLOW:-false}" == "true" ]]; then
    run_tests "Tests lents" "slow" || overall_status=1
else
    echo -e "\n${YELLOW}⏭  Tests lents ignorés (définir RUN_SLOW=true pour les inclure)${NC}"
fi

# 4. Rapport de couverture (optionnel)
if [[ "${WITH_COVERAGE:-false}" == "true" ]]; then
    echo ""
    echo -e "${BLUE}▶ Génération du rapport de couverture${NC}"
    if pytest --cov=security_recon_assistant --cov-report=html --cov-report=term -m "unit or integration"; then
        echo -e "${GREEN}✓ Rapport de couverture généré${NC}"
        echo "   Fon HTML: htmlcov/index.html"
    else
        echo -e "${RED}✗ Échec de la génération du rapport${NC}"
    fi
fi

# Résumé final
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [[ $overall_status -eq 0 ]]; then
    echo -e "${GREEN}✓ Tous les tests ont réussi!${NC}"
else
    echo -e "${RED}✗ Certains tests ont échoué${NC}"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"

exit $overall_status
