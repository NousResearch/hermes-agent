
/* Method object interface */

#ifndef Py_METHODOBJECT_H
#define Py_METHODOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

/* This is about the type 'builtin_function_or_method',
   not Python methods in user-defined classes.  See classobject.h
   for the latter. */

PyAPI_DATA(PyTypeObject) PyCFunction_Type;

#define PyCFunction_CheckExact(op) Py_IS_TYPE(op, &PyCFunction_Type)
#define PyCFunction_Check(op) PyObject_TypeCheck(op, &PyCFunction_Type)

typedef PyObject *(*PyCFunction)(PyObject *, PyObject *);
typedef PyObject *(*_PyCFunctionFast) (PyObject *, PyObject *const *, Py_ssize_t);
typedef PyObject *(*PyCFunctionWithKeywords)(PyObject *, PyObject *,
                                             PyObject *);
typedef PyObject *(*_PyCFunctionFastWithKeywords) (PyObject *,
                                                   PyObject *const *, Py_ssize_t,
                                                   PyObject *);
typedef PyObject *(*PyCMethod)(PyObject *, PyTypeObject *, PyObject *const *,
                               size_t, PyObject *);

// Cast an function to the PyCFunction type to use it with PyMethodDef.
//
// This macro can be used to prevent compiler warnings if the first parameter
// uses a different pointer type than PyObject* (ex: METH_VARARGS and METH_O
// calling conventions).
//
// The macro can also be used for METH_FASTCALL and METH_VARARGS|METH_KEYWORDS
// calling conventions to avoid compiler warnings because the function has more
// than 2 parameters. The macro first casts the function to the
// "void func(void)" type to prevent compiler warnings.
//
// If a function is declared with the METH_NOARGS calling convention, it must
// have 2 parameters. Since the second parameter is unused, Py_UNUSED() can be
// used to prevent a compiler warning. If the function has a single parameter,
// it triggers an undefined behavior when Python calls it with 2 parameters
// (bpo-33012).
#define _PyCFunction_CAST(func) \
    _Py_CAST(PyCFunction, _Py_CAST(void(*)(void), (func)))

PyAPI_FUNC(PyCFunction) PyCFunction_GetFunction(PyObject *);
PyAPI_FUNC(PyObject *) PyCFunction_GetSelf(PyObject *);
PyAPI_FUNC(int) PyCFunction_GetFlags(PyObject *);

Py_DEPRECATED(3.9) PyAPI_FUNC(PyObject *) PyCFunction_Call(PyObject *, PyObject *, PyObject *);

struct PyMethodDef {
    const char  *ml_name;   /* The name of the built-in function/method */
    PyCFunction ml_meth;    /* The C function that implements it */
    int         ml_flags;   /* Combination of METH_xxx flags, which mostly
                               describe the args expected by the C func */
    const char  *ml_doc;    /* The __doc__ attribute, or NULL */
};

/* PyCFunction_New is declared as a function for stable ABI (declaration is
 * needed for e.g. GCC with -fvisibility=hidden), but redefined as a macro
 * that calls PyCFunction_NewEx. */
PyAPI_FUNC(PyObject *) PyCFunction_New(PyMethodDef *, PyObject *);
#define PyCFunction_New(ML, SELF) PyCFunction_NewEx((ML), (SELF), NULL)

/* PyCFunction_NewEx is similar: on 3.9+, this calls PyCMethod_New. */
PyAPI_FUNC(PyObject *) PyCFunction_NewEx(PyMethodDef *, PyObject *,
                                         PyObject *);

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03090000
#define PyCFunction_NewEx(ML, SELF, MOD) PyCMethod_New((ML), (SELF), (MOD), NULL)
PyAPI_FUNC(PyObject *) PyCMethod_New(PyMethodDef *, PyObject *,
                                     PyObject *, PyTypeObject *);
#endif


/* Flag passed to newmethodobject */
/* #define METH_OLDARGS  0x0000   -- unsupported now */
#define METH_VARARGS  0x0001
#define METH_KEYWORDS 0x0002
/* METH_NOARGS and METH_O must not be combined with the flags above. */
#define METH_NOARGS   0x0004
#define METH_O        0x0008

/* METH_CLASS and METH_STATIC are a little different; these control
   the construction of methods for a class.  These cannot be used for
   functions in modules. */
#define METH_CLASS    0x0010
#define METH_STATIC   0x0020

/* METH_COEXIST allows a method to be entered even though a slot has
   already filled the entry.  When defined, the flag allows a separate
   method, "__contains__" for example, to coexist with a defined
   slot like sq_contains. */

#define METH_COEXIST   0x0040

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x030a0000
#  define METH_FASTCALL  0x0080
#endif

/* This bit is preserved for Stackless Python */
#ifdef STACKLESS
#  define METH_STACKLESS 0x0100
#else
#  define METH_STACKLESS 0x0000
#endif

/* METH_METHOD means the function stores an
 * additional reference to the class that defines it;
 * both self and class are passed to it.
 * It uses PyCMethodObject instead of PyCFunctionObject.
 * May not be combined with METH_NOARGS, METH_O, METH_CLASS or METH_STATIC.
 */

#if !defined(Py_LIMITED_API) || Py_LIMITED_API+0 >= 0x03090000
#define METH_METHOD 0x0200
#endif


#ifndef Py_LIMITED_API
#  define Py_CPYTHON_METHODOBJECT_H
#  include "cpython/methodobject.h"
#  undef Py_CPYTHON_METHODOBJECT_H
#endif

#ifdef __cplusplus
}
#endif
#endif /* !Py_METHODOBJECT_H */
