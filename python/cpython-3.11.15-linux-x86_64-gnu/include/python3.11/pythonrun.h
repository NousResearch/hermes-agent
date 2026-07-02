
/* Interfaces to parse and execute pieces of python code */

#ifndef Py_PYTHONRUN_H
#define Py_PYTHONRUN_H
#ifdef __cplusplus
extern "C" {
#endif

PyAPI_FUNC(PyObject *) Py_CompileString(const char *, const char *, int);

PyAPI_FUNC(void) PyErr_Print(void);
PyAPI_FUNC(void) PyErr_PrintEx(int);
PyAPI_FUNC(void) PyErr_Display(PyObject *, PyObject *, PyObject *);


/* Stuff with no proper home (yet) */
PyAPI_DATA(int) (*PyOS_InputHook)(void);

/* Stack size, in "pointers" (so we get extra safety margins
   on 64-bit platforms).  On a 32-bit platform, this translates
   to an 8k margin. */
#define PYOS_STACK_MARGIN 2048

#if defined(WIN32) && !defined(MS_WIN64) && !defined(_M_ARM) && defined(_MSC_VER) && _MSC_VER >= 1300
/* Enable stack checking under Microsoft C */
// When changing the platforms, ensure PyOS_CheckStack() docs are still correct
#define USE_STACKCHECK
#endif

#ifdef USE_STACKCHECK
/* Check that we aren't overflowing our stack */
PyAPI_FUNC(int) PyOS_CheckStack(void);
#endif

#ifndef Py_LIMITED_API
#  define Py_CPYTHON_PYTHONRUN_H
#  include "cpython/pythonrun.h"
#  undef Py_CPYTHON_PYTHONRUN_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_PYTHONRUN_H */
