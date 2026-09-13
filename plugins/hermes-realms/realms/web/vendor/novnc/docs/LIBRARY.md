# Using the noVNC JavaScript library

This document describes how to make use of the noVNC JavaScript library for
integration in your own VNC client application. If you wish to embed the more
complete noVNC application with its included user interface then please see
our [embedding documentation](EMBEDDING.md).

## API

The API of noVNC consists of a single object called `RFB`. The formal
documentation for that object can be found in our [API documentation](API.md).

## Example

noVNC includes a small example application called `vnc_lite.html`. This does
not make use of all the features of noVNC, but is a good start to see how to
do things.

## Conversion of modules

noVNC is written using ECMAScript 6 modules. This is not supported by older
versions of Node.js. To use noVNC with those older versions of Node.js the
library must first be converted.

Fortunately noVNC includes a script to handle this conversion. Please follow
the following steps:

 1. Install Node.js
 2. Run `npm install` in the noVNC directory

The result of the conversion is available in the `lib/` directory.
