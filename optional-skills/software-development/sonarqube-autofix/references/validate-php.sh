#!/usr/bin/env bash
set -euo pipefail

# Validation script for PHP
# Runs PHPStan, PHP CS Fixer and phpunit

echo "Validating PHP..."

# PHPStan
vendor/bin/phpstan analyse

# PHP CS Fixer (formatter)
vendor/bin/php-cs-fixer fix --dry-run

# phpunit (tests with coverage)
vendor/bin/phpunit --coverage-clover=coverage.xml

echo "PHP validation completed."
