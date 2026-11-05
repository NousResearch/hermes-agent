name: Makefile CI

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: configure
      run: ./configure

    - name: Install dependencies
      run: make

    - name: Run check
      run: make check

    - name: Run distcheck
      run: make distcheck
