# Phony targets
.PHONY: help test test-unit test-integration test-all coverage lint format clean install

help:
	@echo "Security Recon Assistant - Makefile"
	@echo ""
	@echo "Commandes disponibles:"
	@echo "  make install        - Installe les dépendances (dev + prod)"
	@echo "  make test          - Lance tous les tests (rapides)"
	@echo "  make test-unit     - Lance seulement les tests unitaires"
	@echo "  make test-integration - Lance les tests d'intégration"
	@echo "  make test-all      - Lance tous les tests (inclut les lents)"
	@echo "  make coverage      - Génère un rapport de couverture HTML"
	@echo "  make lint          - Vérifie le style (flake8, black, isort)"
	@echo "  make format        - Formate le code (black + isort)"
	@echo "  make clean         - Nettoie les fichiers temporaires"
	@echo ""

install:
	pip install -e ".[dev]"

test:
	@echo "▶ Lancement des tests rapides..."
	pytest -v -m "unit or integration" --tb=short

test-unit:
	@echo "▶ Tests unitaires..."
	pytest -v -m "unit" --tb=short

test-integration:
	@echo "▶ Tests d'intégration..."
	pytest -v -m "integration" --tb=short

test-all:
	@echo "▶ Tous les tests (y compris lents)..."
	pytest -v --tb=short

coverage:
	@echo "▶ Génération du rapport de couverture..."
	pytest --cov=security_recon_assistant --cov-report=html --cov-report=term -m "unit or integration"
	@echo "Rapport HTML disponible dans: htmlcov/index.html"

lint:
	@echo "▶ Vérification du style..."
	@command -v flake8 >/dev/null 2>&1 && flake8 security_recon_assistant tests || echo "flake8 non installé, skip"
	@command -v black >/dev/null 2>&1 && black --check security_recon_assistant tests || echo "black non installé, skip"
	@command -v isort >/dev/null 2>&1 && isort --check-only security_recon_assistant tests || echo "isort non installé, skip"

format:
	@echo "▶ Formatage du code..."
	black security_recon_assistant tests
	isort security_recon_assistant tests

clean:
	@echo "▶ Nettoyage..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -f {} + 2>/dev/null || true
	rm -rf dist build 2>/dev/null || true
	@echo "✓ Nettoyage terminé"
