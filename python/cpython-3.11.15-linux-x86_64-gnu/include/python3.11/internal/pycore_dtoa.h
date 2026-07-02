#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

#include "pycore_pymath.h"        // _PY_SHORT_FLOAT_REPR


#if _PY_SHORT_FLOAT_REPR == 1

/* These functions are used by modules compiled as C extension like math:
   they must be exported. */

PyAPI_FUNC(double) _Py_dg_strtod(const char *str, char **ptr);
PyAPI_FUNC(char *) _Py_dg_dtoa(double d, int mode, int ndigits,
                        int *decpt, int *sign, char **rve);
PyAPI_FUNC(void) _Py_dg_freedtoa(char *s);
PyAPI_FUNC(double) _Py_dg_stdnan(int sign);
PyAPI_FUNC(double) _Py_dg_infinity(int sign);

#endif // _PY_SHORT_FLOAT_REPR == 1

#ifdef __cplusplus
}
#endif
