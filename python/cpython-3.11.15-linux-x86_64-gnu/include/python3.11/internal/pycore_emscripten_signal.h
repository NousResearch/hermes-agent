#ifndef Py_EMSCRIPTEN_SIGNAL_H
#define Py_EMSCRIPTEN_SIGNAL_H

#if defined(__EMSCRIPTEN__)

void
_Py_CheckEmscriptenSignals(void);

void
_Py_CheckEmscriptenSignalsPeriodically(void);

#define _Py_CHECK_EMSCRIPTEN_SIGNALS() _Py_CheckEmscriptenSignals()

#define _Py_CHECK_EMSCRIPTEN_SIGNALS_PERIODICALLY() _Py_CheckEmscriptenSignalsPeriodically()

extern int Py_EMSCRIPTEN_SIGNAL_HANDLING;

#else

#define _Py_CHECK_EMSCRIPTEN_SIGNALS()
#define _Py_CHECK_EMSCRIPTEN_SIGNALS_PERIODICALLY()

#endif // defined(__EMSCRIPTEN__)

#endif // ndef Py_EMSCRIPTEN_SIGNAL_H
