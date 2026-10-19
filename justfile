set default-list := true

lint-all: lint-python lint-all-js    

[group('lint python')]
lint-python:
    @ruff check ./

[group('lint python')]
lint-python-fix:
    @ruff check ./ --fix

[group('lint js')]
lint-all-js: lint-web lint-bootstrap-installer lint-desktop lint-tui lint-tests lint-shared

[group('lint js')]
lint-web:
    @cd web && npx eslint ./

[group('lint js')]
lint-bootstrap-installer:
    @cd apps/bootstrap-installer && npx eslint ./

[group('lint js')]
lint-desktop:
    @cd apps/desktop && npx eslint ./

[group('lint js')]
lint-tui:
    @cd ui-tui && npx eslint ./

[group('lint js')]
lint-tests:
    @cd tests-js && npx eslint ./

[group('lint js')]
lint-shared:
    @cd apps/shared && npx eslint ./