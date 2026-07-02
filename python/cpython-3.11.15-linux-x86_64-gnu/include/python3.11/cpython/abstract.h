#ifndef Py_CPYTHON_ABSTRACTOBJECT_H
#  error "this header file must not be included directly"
#endif

/* === Object Protocol ================================================== */

#ifdef PY_SSIZE_T_CLEAN
#  define _PyObject_CallMethodId _PyObject_CallMethodId_SizeT
#endif

/* Convert keyword arguments from the FASTCALL (stack: C array, kwnames: tuple)
   format to a Python dictionary ("kwargs" dict).

   The type of kwnames keys is not checked. The final function getting
   arguments is responsible to check if all keys are strings, for example using
   PyArg_ParseTupleAndKeywords() or PyArg_ValidateKeywordArguments().

   Duplicate keys are merged using the last value. If duplicate keys must raise
   an exception, the caller is responsible to implement an explicit keys on
   kwnames. */
PyAPI_FUNC(PyObject *) _PyStack_AsDict(
    PyObject *const *values,
    PyObject *kwnames);

/* Suggested size (number of positional arguments) for arrays of PyObject*
   allocated on a C stack to avoid allocating memory on the heap memory. Such
   array is used to pass positional arguments to call functions of the
   PyObject_Vectorcall() family.

   The size is chosen to not abuse the C stack and so limit the risk of stack
   overflow. The size is also chosen to allow using the small stack for most
   function calls of the Python standard library. On 64-bit CPU, it allocates
   40 bytes on the stack. */
#define _PY_FASTCALL_SMALL_STACK 5

PyAPI_FUNC(PyObject *) _Py_CheckFunctionResult(
    PyThreadState *tstate,
    PyObject *callable,
    PyObject *result,
    const char *where);

/* === Vectorcall protocol (PEP 590) ============================= */

/* Call callable using tp_call. Arguments are like PyObject_Vectorcall()
   or PyObject_FastCallDict() (both forms are supported),
   except that nargs is plainly the number of arguments without flags. */
PyAPI_FUNC(PyObject *) _PyObject_MakeTpCall(
    PyThreadState *tstate,
    PyObject *callable,
    PyObject *const *args, Py_ssize_t nargs,
    PyObject *keywords);

#define PY_VECTORCALL_ARGUMENTS_OFFSET \
    (_Py_STATIC_CAST(size_t, 1) << (8 * sizeof(size_t) - 1))

static inline Py_ssize_t
PyVectorcall_NARGS(size_t n)
{
    return n & ~PY_VECTORCALL_ARGUMENTS_OFFSET;
}

PyAPI_FUNC(vectorcallfunc) PyVectorcall_Function(PyObject *callable);

PyAPI_FUNC(PyObject *) PyObject_Vectorcall(
    PyObject *callable,
    PyObject *const *args,
    size_t nargsf,
    PyObject *kwnames);

// Backwards compatibility aliases for API that was provisional in Python 3.8
#define _PyObject_Vectorcall PyObject_Vectorcall
#define _PyObject_VectorcallMethod PyObject_VectorcallMethod
#define _PyObject_FastCallDict PyObject_VectorcallDict
#define _PyVectorcall_Function PyVectorcall_Function
#define _PyObject_CallOneArg PyObject_CallOneArg
#define _PyObject_CallMethodNoArgs PyObject_CallMethodNoArgs
#define _PyObject_CallMethodOneArg PyObject_CallMethodOneArg

/* Same as PyObject_Vectorcall except that keyword arguments are passed as
   dict, which may be NULL if there are no keyword arguments. */
PyAPI_FUNC(PyObject *) PyObject_VectorcallDict(
    PyObject *callable,
    PyObject *const *args,
    size_t nargsf,
    PyObject *kwargs);

/* Call "callable" (which must support vectorcall) with positional arguments
   "tuple" and keyword arguments "dict". "dict" may also be NULL */
PyAPI_FUNC(PyObject *) PyVectorcall_Call(PyObject *callable, PyObject *tuple, PyObject *dict);

// Same as PyObject_Vectorcall(), except without keyword arguments
PyAPI_FUNC(PyObject *) _PyObject_FastCall(
    PyObject *func,
    PyObject *const *args,
    Py_ssize_t nargs);

PyAPI_FUNC(PyObject *) PyObject_CallOneArg(PyObject *func, PyObject *arg);

PyAPI_FUNC(PyObject *) PyObject_VectorcallMethod(
    PyObject *name, PyObject *const *args,
    size_t nargsf, PyObject *kwnames);

static inline PyObject *
PyObject_CallMethodNoArgs(PyObject *self, PyObject *name)
{
    size_t nargsf = 1 | PY_VECTORCALL_ARGUMENTS_OFFSET;
    return PyObject_VectorcallMethod(name, &self, nargsf, _Py_NULL);
}

static inline PyObject *
PyObject_CallMethodOneArg(PyObject *self, PyObject *name, PyObject *arg)
{
    PyObject *args[2] = {self, arg};
    size_t nargsf = 2 | PY_VECTORCALL_ARGUMENTS_OFFSET;
    assert(arg != NULL);
    return PyObject_VectorcallMethod(name, args, nargsf, _Py_NULL);
}

PyAPI_FUNC(PyObject *) _PyObject_CallMethod(PyObject *obj,
                                            PyObject *name,
                                            const char *format, ...);

/* Like PyObject_CallMethod(), but expect a _Py_Identifier*
   as the method name. */
PyAPI_FUNC(PyObject *) _PyObject_CallMethodId(PyObject *obj,
                                              _Py_Identifier *name,
                                              const char *format, ...);

PyAPI_FUNC(PyObject *) _PyObject_CallMethodId_SizeT(PyObject *obj,
                                                    _Py_Identifier *name,
                                                    const char *format,
                                                    ...);

PyAPI_FUNC(PyObject *) _PyObject_CallMethodIdObjArgs(
    PyObject *obj,
    _Py_Identifier *name,
    ...);

static inline PyObject *
_PyObject_VectorcallMethodId(
    _Py_Identifier *name, PyObject *const *args,
    size_t nargsf, PyObject *kwnames)
{
    PyObject *oname = _PyUnicode_FromId(name); /* borrowed */
    if (!oname) {
        return _Py_NULL;
    }
    return PyObject_VectorcallMethod(oname, args, nargsf, kwnames);
}

static inline PyObject *
_PyObject_CallMethodIdNoArgs(PyObject *self, _Py_Identifier *name)
{
    size_t nargsf = 1 | PY_VECTORCALL_ARGUMENTS_OFFSET;
    return _PyObject_VectorcallMethodId(name, &self, nargsf, _Py_NULL);
}

static inline PyObject *
_PyObject_CallMethodIdOneArg(PyObject *self, _Py_Identifier *name, PyObject *arg)
{
    PyObject *args[2] = {self, arg};
    size_t nargsf = 2 | PY_VECTORCALL_ARGUMENTS_OFFSET;
    assert(arg != NULL);
    return _PyObject_VectorcallMethodId(name, args, nargsf, _Py_NULL);
}

PyAPI_FUNC(int) _PyObject_HasLen(PyObject *o);

/* Guess the size of object 'o' using len(o) or o.__length_hint__().
   If neither of those return a non-negative value, then return the default
   value.  If one of the calls fails, this function returns -1. */
PyAPI_FUNC(Py_ssize_t) PyObject_LengthHint(PyObject *o, Py_ssize_t);

/* === Sequence protocol ================================================ */

/* Assume tp_as_sequence and sq_item exist and that 'i' does not
   need to be corrected for a negative index. */
#define PySequence_ITEM(o, i)\
    ( Py_TYPE(o)->tp_as_sequence->sq_item(o, i) )

#define PY_ITERSEARCH_COUNT    1
#define PY_ITERSEARCH_INDEX    2
#define PY_ITERSEARCH_CONTAINS 3

/* Iterate over seq.

   Result depends on the operation:

   PY_ITERSEARCH_COUNT:  return # of times obj appears in seq; -1 if
     error.
   PY_ITERSEARCH_INDEX:  return 0-based index of first occurrence of
     obj in seq; set ValueError and return -1 if none found;
     also return -1 on error.
   PY_ITERSEARCH_CONTAINS:  return 1 if obj in seq, else 0; -1 on
     error. */
PyAPI_FUNC(Py_ssize_t) _PySequence_IterSearch(PyObject *seq,
                                              PyObject *obj, int operation);

/* === Mapping protocol ================================================= */

PyAPI_FUNC(int) _PyObject_RealIsInstance(PyObject *inst, PyObject *cls);

PyAPI_FUNC(int) _PyObject_RealIsSubclass(PyObject *derived, PyObject *cls);

PyAPI_FUNC(char *const *) _PySequence_BytesToCharpArray(PyObject* self);

PyAPI_FUNC(void) _Py_FreeCharPArray(char *const array[]);

/* For internal use by buffer API functions */
PyAPI_FUNC(void) _Py_add_one_to_index_F(int nd, Py_ssize_t *index,
                                        const Py_ssize_t *shape);
PyAPI_FUNC(void) _Py_add_one_to_index_C(int nd, Py_ssize_t *index,
                                        const Py_ssize_t *shape);

/* Convert Python int to Py_ssize_t. Do nothing if the argument is None. */
PyAPI_FUNC(int) _Py_convert_optional_to_ssize_t(PyObject *, void *);

/* Same as PyNumber_Index but can return an instance of a subclass of int. */
PyAPI_FUNC(PyObject *) _PyNumber_Index(PyObject *o);
